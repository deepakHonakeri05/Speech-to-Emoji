import speech_recognition as sr

r = sr.Recognizer()

with sr.Microphone() as source:
	print("Speak Anything : ")
	audio = r.listen(source)

	try:
		text = r.recognize_google(audio)
		print("Output : " + str(text))
	except:
		print("Couldn't recognize your voice")