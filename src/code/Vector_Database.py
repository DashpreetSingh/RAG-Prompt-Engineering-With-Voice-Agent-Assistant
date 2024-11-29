from dotenv import load_dotenv
import os
import openai
import faiss
import pickle
import numpy as np
from src.code.Preprocess import Preprocess

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')


class VectorDatabase:
    def __init__(self):
        self.VD_path = os.path.join("Vector_Storage")
        self.run = Preprocess()

    def get_openai_embeddings(self, text_list):
        embeddings = []
        try:
            # Request embeddings from OpenAI for each text in the list
            for text in text_list:
                response = openai.Embedding.create(
                    model="text-embedding-ada-002",
                    input=text
                )
                embeddings.append(response['data'][0]['embedding'])
        except Exception as e:
            print(f"Error in getting embeddings from OpenAI: {e}")
            raise  # Re-raise to handle upstream
        return embeddings

    def store_text_as_vectors(self):
        # Ensure the output directory exists
        os.makedirs(self.VD_path, exist_ok=True)

        # Get cleaned text data
        text_data = self.run.punctuationRemove()
        if not isinstance(text_data, list):
            raise ValueError("punctuationRemove() must return a list of strings.")

        # Get OpenAI embeddings for the text data
        vectors = self.get_openai_embeddings(text_data)

        # Convert the list of vectors to a NumPy array
        vectors = np.array(vectors).astype(np.float32)

        # Initialize FAISS index
        dimension = len(vectors[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(vectors)

        # File paths for storing the index and pickle file
        faiss_index_path = os.path.join(self.VD_path, "vector_store.index")
        pickle_file_path = os.path.join(self.VD_path, "vector_store_data.pkl")

        # Save the FAISS index and text data
        faiss.write_index(index, faiss_index_path)
        with open(pickle_file_path, "wb") as f:
            pickle.dump(text_data, f)

        print(f"Vectors and extracted text data stored successfully in folder: {self.VD_path}")


# if __name__ == "__main__":
#     VD = VectorDatabase()
#     VD.store_text_as_vectors()
