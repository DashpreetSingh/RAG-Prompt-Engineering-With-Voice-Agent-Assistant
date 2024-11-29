import openai
import os
import pyttsx3
from dotenv import load_dotenv
import speech_recognition as sr
from src.code.Prompt_Engineering import QueryHandler

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


class VoiceAgent:
    def __init__(self):
        self.speaker = pyttsx3.init()
        self.query_handler = QueryHandler()
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()


    def listen_for_command(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            print("Say something i am Listerning...")
            audio = self.recognizer.listen(source)

        try:
            print("Recognizing...")
            transcript = self.recognizer.recognize_google(audio)
            print(f"Recognized command: {transcript}")
            return transcript
        except sr.UnknownValueError:
            # print("Google Speech Recognition could not understand the audio")
            return ""
        except sr.RequestError as e:
            print(f"Google Speech Recognition request failed: {e}")
            return ""

    def speak_response(self, response):
        """Convert text response into speech using pyttsx3"""
        print(f"Speaking response: {response}")
        self.speaker.say(response)
        self.speaker.runAndWait()


if __name__ == "__main__":
    try:
        voice_agent = VoiceAgent()

        while True:
            # Step 1: Listen for a command
            prompt = voice_agent.listen_for_command()

            if prompt:
                if "stop" in prompt.lower() or "exit" in prompt.lower():
                    print("Stopping the Voice Agent")
                    voice_agent.speak_response("Good Bye!")
                    break

                # Step 2: Generate a response based on the prompt
                response = voice_agent.query_handler.generate_response(prompt, k=3)

                # Step 3: Speak the response aloud
                voice_agent.speak_response(response)

    except Exception as main_error:
        print(f"An error occurred: {main_error}")
