import faiss
import pickle
import openai
import os
import numpy as np
from dotenv import load_dotenv


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class QueryHandler:
    def __init__(self):
        self.vector_storage_folder = os.path.join("Vector_Storage")

        faiss_index_path = os.path.join(self.vector_storage_folder, "vector_store.index")
        text_data_path = os.path.join(self.vector_storage_folder, "vector_store_data.pkl")

        self.index = faiss.read_index(faiss_index_path)
        with open(text_data_path, "rb") as f:
            self.extracted_text = pickle.load(f)

    def generate_embedding(self, text):
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return np.array(response['data'][0]['embedding'], dtype=np.float32)

    def retrieve_relevant_data(self, query, k=3):
        """
        Retrieve the top-k most relevant text data for a given query.

        :param query: Query string.
        :param k: Number of relevant results to retrieve.
        :return: List of relevant text data.
        """
        try:
            query_vector = self.generate_embedding(query)
            query_vector = np.expand_dims(query_vector, axis=0)  # Reshape for FAISS
            distances, indices = self.index.search(query_vector, k)
            return [self.extracted_text[i] for i in indices[0] if i != -1]
        except Exception as e:
            print(f"Error retrieving relevant data for query '{query}': {e}")

    def generate_response(self, prompt, k=3):
        relevant_contexts = []
        response_list = []
        try:
            relevant_contexts.extend(self.retrieve_relevant_data(prompt, k=k))

            # Combine retrieved contexts
            relevant_data = "\n".join(relevant_contexts[:k])  # Limit to k contexts

            final_prompt = (
                f"Context:\n{relevant_data}\n\n"
                f"User Query:\n{prompt}\n\n"
                """Instruction: Based on the given context, generate a response about the Ketto donation campaigns.,
                """
            )

            response = openai.ChatCompletion.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": "You are a donation campaign assistant focused on encouraging users to support noble causes via Ketto SIP."},
                    {"role": "user", "content": final_prompt}
                ],
                max_tokens=400,
                temperature=0.5
            )
            response_list.append(response['choices'][0]['message']['content'].strip())
        except Exception as e:
            print(f"Error generating response': {e}")

        result = "\n".join(response_list)
        response = self.remove_special_characters(result)
        return response

    def remove_special_characters(self, response):
        """
        Removes '*' and '#' characters from the given response string.

        :param response: The input string to clean.
        :return: The cleaned string with '*' and '#' removed.
        """
        response = response.replace('*', '').replace('#', '').replace('/', '')
        return response

if __name__ == "__main__":
    try:
        handler = QueryHandler()
        #prompting
        # prompt = "How to start conversation with donar?"
        prompt = "what is women pitch?"
        print("prompt: ",prompt)
        response = handler.generate_response(prompt, k=3)
        print("response: ", response)
    except Exception as main_error:
        print(f"An error occurred: {main_error}")

