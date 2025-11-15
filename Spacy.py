import triton_python_backend_utils as pb_utils
import numpy as np
import logging
import spacy
from pathlib import Path
from packages.finlogging import get_log
from packages.funcs import read_json
from dateutil.parser import parse
from typing import Optional, List, Tuple
import os
import re
import psutil


class SpacyModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """'initialize' is called only once when the model is being loaded.
        Implementing 'initialize' function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
            Both keys and values are strings. The dictionary keys and values are:
            * model_config: A JSON string containing the model configuration
            * model_instance_kind: A string containing model instance kind
            * model_instance_device_id: A string containing model instance device ID
            * model_repository: Model repository path
            * model_version: Model version
        """

        self.logger = pb_utils.Logger
        MODEL_PATH = str(Path(__file__).parent.parent.joinpath(f"models/{args.get('model_name', 'us_spacy_ner')}/{args.get('model_version', '1')}"))
        self.logger.log_info(get_log(logging.INFO, f"MODEL NAME: {args.get('model_name', 'us_spacy_ner')}"))
        print(f"MODEL NAME: {args.get('model_name', 'us_spacy_ner')}")
        self.logger.log_info(get_log(logging.INFO, f"MODEL PATH: {MODEL_PATH}"))
        print(f"MODEL PATH: {MODEL_PATH}")
        
        # 1. Load the base model
        self.nlp = spacy.load(MODEL_PATH)
        
        # 2. === NEW: Add doc_cleaner to the pipeline ===
        # This will run last and remove the large .tensor attribute
        if 'doc_cleaner' not in self.nlp.pipe_names:
            self.nlp.add_pipe('doc_cleaner', last=True)
            print("Successfully added 'doc_cleaner' to the pipeline.")
        else:
            print("'doc_cleaner' is already in the pipeline.")

        # 3. === NEW: Print all components to confirm ===
        print(f"Loaded pipeline components: {self.nlp.pipe_names}")
        
        self.logger.log_info(get_log(logging.INFO, "Initialized..."))
        print("Initialized...")
        
        self.descriptor_arr = os.environ.get('DESCRIPTORS_TO_REMOVE', '').split(',')
        self.premodel_removal_arr = read_json(str(Path(__file__).parent.parent.joinpath(f"models/{args.get('model_name', 'us_spacy_ner')}/premodel_removal.json")))
        self.postmodel_removal_arr = read_json(str(Path(__file__).parent.parent.joinpath(f"models/{args.get('model_name', 'us_spacy_ner')}/postmodel_removal.json")))
        self.month_abbr_arr = read_json(str(Path(__file__).parent.parent.joinpath(f"models/{args.get('model_name', 'us_spacy_ner')}/month_abbr.json")))
        self.ignore_business_entity_arr = read_json(str(Path(__file__).parent.parent.joinpath(f"models/{args.get('model_name', 'us_spacy_ner')}/ignore_business_entity.json")))
        self.remove_descriptors_vectorized = np.vectorize(self.remove_descriptors_from_entity, otypes=['O'])
        self.strip_chars_from_right_vectorized = np.vectorize(self.strip_chars_from_right_entity, otypes=['O'])
        self.regex_patterns = self.load_regex_patterns(args)
        self.regex_filter_ent = read_json(str(Path(__file__).parent.parent.joinpath(f"models/{args.get('model_name', 'us_spacy_ner')}/regex_filters.json")))
        
        self.use_memory_zones = args.get('use_memory_zones', True)
        
        # 4. === NEW: Add flag for select_pipes ===
        self.use_select_pipes = False # Default to off, main.py will turn it on
        
        self.batch_size = int(args.get('batch_size', 100))
        self.process = psutil.Process(os.getpid())
        self.memory_stats = []
        print(f"Memory Zones: {'ENABLED' if self.use_memory_zones else 'DISABLED'}")
        print(f"Batch Size: {self.batch_size}")

    def load_regex_patterns(self, args):
        """Load regex patterns and capturing groups from a JSON file."""
        """Returns a list of tuples, where each tuple contains a compiled regex pattern and its corresponding capturing group index."""
        regex_patterns_path = str(Path(__file__).parent.parent.joinpath(f"models/{args.get('model_name', 'us_spacy_ner')}/regex_patterns.json"))
        try:
            patterns = read_json(regex_patterns_path)
            return [(re.compile(pattern['pattern']), pattern['capturing_group']) for pattern in patterns]
        except Exception as e:
            self.logger.log_warn(get_log(logging.WARNING, f"Failed to load regex patterns: {e}"))
            print(f"WARNING: Failed to load regex patterns: {e}")
            return []

    def get_memory_stats(self, label):
        mem_info = self.process.memory_info()
        return {
            'label': label,
            'rss_mb': mem_info.rss / 1024 / 1024,
            'vocab_size': len(self.nlp.vocab),
            'stringstore_size': len(self.nlp.vocab.strings)
        }

    def log_memory_stats(self, stats):
        log_msg = (f"Memory [{stats['label']}]: RSS={stats['rss_mb']:.2f}MB, "
                   f"Vocab={stats['vocab_size']}, StringStore={stats['stringstore_size']}")
        self.logger.log_info(get_log(logging.INFO, log_msg))
        print(log_msg)

    def extract_results(self, docs, descriptions, memos):
        entities = []
        all_labels = []
        all_extracted_texts = []
        all_entity_ids = []

        max_len = 0
        for doc, description, memo in zip(docs, descriptions, memos):

            labels = []
            extracted_texts = []
            entity_ids = []

            sub_entities = ""

            # Get flag for IGNORE_ALL
            # Note: We must check all tags before deciding if we need to "ignore_all" ents, so this iteration is
            #       unavoidable, but we make it as efficient as possible by only checking the label
            ignore_all_flag = False
            for ent in doc.ents:
                if ent.label_ == "IGNORE_ALL":
                    ignore_all_flag = True
                    break  # One IGNORE_ALL entity label is enough, we can stop checking

            # Concatenate description and memo
            desc_memo = self.combine_desc_memo(desc=description, memo=memo)

            # Go through ents and exclude/clean ents based on postprocessing logic
            for ent in doc.ents:

                # Clean the entity according to post-processing logic
                clean_ent = self.clean_ent(ent.text)

                # If the entity is not valid then we skip it
                if not self.is_valid_ent(ent.label_, clean_ent, ignore_all_flag=ignore_all_flag):
                    continue

                labels.append(ent.label_)
                extracted_texts.append(clean_ent)
                entity_ids.append(ent.ent_id_)

                e = f"{ent.text}-{ent.label_}-{ent.ent_id_}| "
                sub_entities += e

        regex_flag = False

        if len(labels) == 0:  # Case 1: No entities found
            regex_flag = True
        else:
            # Case 2: Has PLATFORMS with specific entity IDs and no Business Entity in label
            if ("PLATFORMS" in labels) and not("Business entity" in labels) and any(id in entity_ids for id in self.regex_filter_ent):
                regex_flag = True
            # Case 3: Missing all three specific labels
            elif not any(label in labels for label in ["PLATFORMS", "PAYMENT PROCESSORS", "Business entity"]):
                regex_flag = True

        if regex_flag:
            # if regex condition is satisfied

            # desc memo passed to regex pattern function
            text_ = self.apply_regex_patterns_if_no_business_entity(desc_memo, "", "")
            # Post processing functions applied on regex output
            clean_ent = self.clean_ent(text_)
            # if clean ent is not blank or None after post processing cleaning
            if clean_ent:
                extracted_texts.append(clean_ent)
                labels.append("Business entity")
                entity_ids.append("")
                sub_entities += f"{clean_ent}-Business entity|| "

        max_len = len(labels) if len(labels) > max_len else max_len
        entities.append(sub_entities)
        all_labels.append(labels)
        all_extracted_texts.append(extracted_texts)
        all_entity_ids.append(entity_ids)

        return (entities, ) + (self.normalize_lengths(
            max_len, ["", "", "", -1],
            all_labels, all_extracted_texts, all_entity_ids
        ))

    def apply_regex_patterns_if_no_business_entity(self, desc_memo, text, label):
        """Apply regex patterns to the desc_memo if no Business entity found"""
        try:
            if label != "Business entity":
                for pattern, capturing_group in self.regex_patterns:
                    match_obj = pattern.search(desc_memo)
                    if match_obj:
                        extracted_text = match_obj.group(capturing_group)
                        if extracted_text:
                            return extracted_text, "Business entity"  # Extract the string from the specified capturing group
        except Exception as e:
            return text, label
        return text, label

    def normalize_lengths(self, max_len, defaults=[], *args):
        ret_args = []
        defaults = [""]*len(args) if not defaults else defaults
        for idx, arg in enumerate(args):
            ret_arg = []
            for each_arg_list in arg:
                res_list = each_arg_list + [defaults[idx]]*(max_len - len(each_arg_list))
                ret_arg.append(res_list)
            ret_args.append(ret_arg)
        return tuple(ret_args)

    def combine_desc_memo(self, desc: Optional[str] = "", memo: Optional[str] = "") -> str:
        """Puts the description and memo together into one string to be put into the ML model."""
        return str(desc) + " " + str(memo)

    def clean_text(self, text: str, removal_arr: list) -> str:
        """Traditional text cleaning like removing special characters"""
        for txt in removal_arr:
            text = text.replace(txt, " ")
        
        # remove extra white space. Added because OS does this during tests, even though it isn't in their real preprocessor.py
        text = " ".join(text.split()).strip()
        return text

    def preprocess_input(self, descriptions: List[str], memos: List[str]) -> List:
        texts = [
            self.combine_desc_memo(desc=description, memo=memo)
            for description, memo in zip(descriptions, memos)
        ]
        texts = [self.clean_text(text, self.premodel_removal_arr) for text in texts]
        return texts

    def is_valid_ent(self, label, ent, ignore_all_flag=False) -> bool:
        if not ent:
            return False
        
        # IGNORE_ALL entity check
        if ignore_all_flag and label not in ["PAYMENT TYPES", "PLATFORMS"]:
            return False
        
        # Check if the Business entity is in the ignore list
        if label == "Business entity":
            if ent.lower() in self.ignore_business_entity_arr:
                self.logger.log_info(get_log(logging.INFO, f"Ignored entity: {ent} (label: {label})"))
                return False
        
        # In entities extracted from the entity recognizer, if the entity text has
        # <2 alphabetical characters, exclude that entity
        if not label == "Business entity lookup":
            if sum([char.isalpha() for char in ent]) < 2:
                return False
        
        return True  # If all exclusion conditions were passed, entity is valid

    def clean_ent(self, entity: str) -> str:
        clean_entity = self.clean_x(entity)
        clean_entity = self.clean_text(clean_entity, self.postmodel_removal_arr)
        clean_entity = self.remove_descriptors_from_entity(clean_entity)
        clean_entity = self.remove_duplicate_tokens(clean_entity)
        clean_entity = self.strip_chars_from_right_entity(clean_entity)
        clean_entity = self.remove_month_date_year_from_entity(clean_entity, self.month_abbr_arr)
        clean_entity = self.camel_case_output(clean_entity)
        return clean_entity

    def clean_x(self, entity):
        """Removes occurrences of xxxx+ pattern (at least 4 in a row)"""
        entity = re.sub('xxxx+', ' ', entity, flags=re.IGNORECASE)
        return entity.strip()

    def strip_chars_from_right_entity(self, entity: str) -> str:
        # Strip from right side non-alphanumeric characters
        while len(entity) > 0 and not entity[-1].isalnum():
            entity = entity[:-1]
        return entity

    def remove_descriptors_from_entity(self, entity_name):
        # remove specified entity name descriptors
        split_entity_names = entity_name.split()
        filtered_entity_names = [
            str(split_entity_name).lower().replace('&amp;', '&') if '&amp;' in str(split_entity_name).lower() else split_entity_name
            for split_entity_name in split_entity_names if not self.is_descriptor(split_entity_name)
        ]
        cleaned_entity = ' '.join(filtered_entity_names)
        return cleaned_entity

    def remove_duplicate_tokens(self, entity_name):
        """Removes repeated tokens from the entity name (i.e. Access Pay Access Pay -> Access Pay)"""
        used_tokens = set()
        final_tokens = []
        for token in entity_name.split():
            if token.lower() not in used_tokens:
                final_tokens.append(token)
                used_tokens.add(token.lower())
        return " ".join(final_tokens)

    def remove_month_date_year_from_entity(self, entity_name: str, month_abbr_arr: list) -> str:
        """Removes date year and month from extracted entity name (i.e. EDUCATION DETTS625JAN2023 -> EDUCATION DETTS)"""
        np_lwr = entity_name.lower()
        for mnth in month_abbr_arr:
            if mnth in np_lwr:
                try:
                    return " ".join([i.strip() for i in parse(entity_name, fuzzy=True, fuzzy_with_tokens=True)[1] if i != ' '])
                except Exception as e:
                    return entity_name
        return entity_name

    def camel_case_output(self, sentence: str) -> str:
        """Camel cases the entity output (i.e. "STAR OSCO" -> "Star Osco", "SHAWS.COM" -> "Shaws.com", "bell's market" -> "Bell's Market")"""
        try:
            sentence = str(sentence)
            camel_case_words_list = [word.capitalize() for word in sentence.split()]
            return " ".join(camel_case_words_list)
        except Exception as e:
            return sentence

    def is_descriptor(self, word):
        # Check if the word is a descriptor
        return re.sub(r'\b[^\w\s,]\b', '', word.lower()) in map(str.lower, self.descriptor_arr)

    def test_isolation(self, texts: List[str], batch_size: int = None) -> Tuple[List[dict], List[dict]]:
        if batch_size is None:
            batch_size = self.batch_size
        
        print(f"\n{'='*70}")
        print(f"STARTING ISOLATION TEST")
        print(f"Total texts: {len(texts)}, Batch size: {batch_size}")
        print(f"Test Config: MemoryZone={self.use_memory_zones}, SelectPipes={self.use_select_pipes}")
        print(f"{'='*70}\n")
        
        results = []
        memory_stats = []
        
        initial_stats = self.get_memory_stats("initial")
        memory_stats.append(initial_stats)
        self.log_memory_stats(initial_stats)
        
        num_batches = (len(texts) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(texts), batch_size):
            batch_num = batch_idx // batch_size
            batch_texts = texts[batch_idx:batch_idx + batch_size]
            
            print(f"\nProcessing batch {batch_num + 1}/{num_batches} (samples {batch_idx} to {batch_idx + len(batch_texts) - 1})...")
            
            before_stats = self.get_memory_stats(f"before_batch_{batch_num}")
            memory_stats.append(before_stats)
            self.log_memory_stats(before_stats)
            
            if self.use_memory_zones:
                with self.nlp.memory_zone():
                    
                    # === NEW: Add select_pipes logic ===
                    if self.use_select_pipes:
                        with self.nlp.select_pipes(enable=["entity_ruler", "ner"]):
                            docs = list(self.nlp.pipe(batch_texts))
                    else:
                        docs = list(self.nlp.pipe(batch_texts))
                    # === End of new logic ===
                        
                    for doc in docs:
                        doc_results = []
                        for ent in doc.ents:
                            doc_results.append({
                                'text': ent.text,
                                'label': ent.label_,
                                'start': ent.start_char,
                                'end': ent.end_char,
                                'ent_id': ent.ent_id_
                            })
                        results.append(doc_results)
            else:
                docs = list(self.nlp.pipe(batch_texts))
                for doc in docs:
                    doc_results = []
                    for ent in doc.ents:
                        doc_results.append({
                            'text': ent.text,
                            'label': ent.label_,
                            'start': ent.start_char,
                            'end': ent.end_char,
                            'ent_id': ent.ent_id_
                        })
                    results.append(doc_results)
            
            after_stats = self.get_memory_stats(f"after_batch_{batch_num}")
            memory_stats.append(after_stats)
            self.log_memory_stats(after_stats)
            
            growth_mb = after_stats['rss_mb'] - before_stats['rss_mb']
            growth_vocab = after_stats['vocab_size'] - after_stats['vocab_size']
            print(f"  Batch growth: +{growth_mb:.2f}MB, +{growth_vocab} vocab entries")
        
        final_stats = self.get_memory_stats("final")
        memory_stats.append(final_stats)
        self.log_memory_stats(final_stats)
        
        print(f"\n{'='*70}")
        print(f"ISOLATION TEST COMPLETED")
        print(f"{'='*70}\n")
        
        return results, memory_stats

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []
        request_sizes = []
        input_all = []
        descriptions_all = []
        memos_all = []

        # Every Python backend must iterate through list of requests and create
        # an instance of pb_utils.InferenceResponse class for each of them. You
        # should avoid storing any of the input Tensors in the class attributes
        # as they will be overridden in subsequent inference requests. You can
        # make a copy of the underlying NumPy array and store it if it is
        # required.
        for request in requests:
            # Perform inference on the request and append it to responses list...

            # Parse input bytes into list of string
            descriptions = pb_utils.get_input_tensor_by_name(request, "description")
            descriptions_ls = [x[0].decode('utf-8') for x in descriptions.as_numpy()]

            memos = pb_utils.get_input_tensor_by_name(request, "memo")
            memos_ls = [x[0].decode('utf-8') for x in memos.as_numpy()]

            request_sizes.append(len(descriptions_ls))

            inputs = self.preprocess_input(descriptions_ls, memos_ls)
            input_all.extend(inputs)
            descriptions_all.extend(descriptions_ls)
            memos_all.extend(memos_ls)

        self.logger.log_info(get_log(logging.INFO, f"request_sizes = {request_sizes}"))
        print(f"\nrequest_sizes = {request_sizes}")
        
        print(f"\n{'='*70}")
        print(f"STARTING FULL PIPELINE EXECUTION")
        print(f"Total samples: {len(input_all)}, Batch size: {self.batch_size}")
        print(f"Test Config: MemoryZone={self.use_memory_zones}, SelectPipes={self.use_select_pipes}")
        print(f"{'='*70}\n")
        
        initial_stats = self.get_memory_stats("initial")
        self.memory_stats.append(initial_stats)
        self.log_memory_stats(initial_stats)
        
        all_labels = []
        all_extracted_texts = []
        all_ent_ids = []
        
        num_batches = (len(input_all) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(0, len(input_all), self.batch_size):
            batch_num = batch_idx // self.batch_size
            batch_inputs = input_all[batch_idx:batch_idx + self.batch_size]
            batch_descriptions = descriptions_all[batch_idx:batch_idx + self.batch_size]
            batch_memos = memos_all[batch_idx:batch_idx + self.batch_size]
            
            print(f"\nProcessing batch {batch_num + 1}/{num_batches} (samples {batch_idx} to {batch_idx + len(batch_inputs) - 1})...")
            
            before_stats = self.get_memory_stats(f"before_batch_{batch_num}")
            self.memory_stats.append(before_stats)
            self.log_memory_stats(before_stats)
            
            if self.use_memory_zones:
                with self.nlp.memory_zone():
                    
                    # === NEW: Add select_pipes logic ===
                    if self.use_select_pipes:
                        print("Running with: select_pipes(enable=['entity_ruler', 'ner'])")
                        with self.nlp.select_pipes(enable=["entity_ruler", "ner"]):
                            result_docs = list(self.nlp.pipe(batch_inputs))
                    else:
                        print("Running with: all pipes")
                        result_docs = list(self.nlp.pipe(batch_inputs))
                    # === End of new logic ===
                        
                    _, batch_labels, batch_extracted_texts, batch_ent_ids = self.extract_results(
                        result_docs, batch_descriptions, batch_memos
                    )
            else:
                print("Running with: all pipes (no zone)")
                result_docs = list(self.nlp.pipe(batch_inputs))
                _, batch_labels, batch_extracted_texts, batch_ent_ids = self.extract_results(
                    result_docs, batch_descriptions, batch_memos
                )
            
            all_labels.extend(batch_labels)
            all_extracted_texts.extend(batch_extracted_texts)
            all_ent_ids.extend(batch_ent_ids)
            
            after_stats = self.get_memory_stats(f"after_batch_{batch_num}")
            self.memory_stats.append(after_stats)
            self.log_memory_stats(after_stats)
            
            growth_mb = after_stats['rss_mb'] - before_stats['rss_mb']
            growth_vocab = after_stats['vocab_size'] - before_stats['vocab_size']
            print(f"  Batch growth: +{growth_mb:.2f}MB, +{growth_vocab} vocab entries")
        
        final_stats = self.get_memory_stats("final")
        self.memory_stats.append(final_stats)
        self.log_memory_stats(final_stats)
        
        print(f"\n{'='*70}")
        print(f"FULL PIPELINE EXECUTION COMPLETED")
        print(f"{'='*70}\n")

        # Vectorize for tensor output
        ret_labels = np.array(all_labels, dtype='|S0')
        ret_extracted_texts = np.array(all_extracted_texts, dtype=object)
        ret_ent_ids = np.array(all_ent_ids, dtype='|S0')

        start_idx = 0
        for request_size in request_sizes:
            end_idx = start_idx + request_size

            # Add tokens to Inference Response
            out_labels = pb_utils.Tensor("label", ret_labels[start_idx:end_idx])
            out_extracted_texts = pb_utils.Tensor("extractedText", ret_extracted_texts[start_idx:end_idx])
            out_ent_ids = pb_utils.Tensor("entityId", ret_ent_ids[start_idx:end_idx])

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    out_labels,
                    out_extracted_texts,
                    out_ent_ids
                ]
            )
            start_idx = end_idx
            responses.append(inference_response)

        # You must return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        self.logger.log_info(get_log(logging.INFO, "Cleaning up..."))
        print("Cleaning up...")
