Proiectul ăsta e un translator real-time pentru limbajul semnelor (ASL). Ideea e simplă, dar cu impact: folosești o cameră web ca să citești semnele făcute cu mâna, iar codul le transformă în cuvinte și le rostește cu voce tare. Scopul e clar – să ajute persoanele cu deficiențe de auz sau vorbire să comunice natural cu oricine, chiar dacă persoana din fața lor nu știe deloc limbajul semnelor.

Pe partea tehnică, am scris totul în Python pentru că e cel mai eficient ecosistem pentru Machine Learning și Computer Vision. Arhitectura are două piese mari:

1. Detectarea și controlul (Computer Vision)
Folosesc OpenCV pentru captarea imaginii și MediaPipe pentru tracking-ul mâinii. MediaPipe e super rapid: îmi găsește instant coordonatele degetelor și îmi face crop exact pe zona în care se află mâna. Pe lângă asta, am mapat niște gesturi logice folosind geometria degetelor pentru sistemul de control: degetul în jos face "delete", pumnul dă "space", iar semnul păcii (V) declanșează vorbirea propoziției.

2. Creierul de Machine Learning (CNN)
Aici intră în joc fișierele de training. Pentru date, am mers pe o sugestie de la băieții de la Sigmoid și am luat dataset-ul Sign Language MNIST de pe Kaggle. E practic un standard bun, având mii de imagini convertite în valori de pixeli în acele fișiere CSV.

În train.py, am construit o Rețea Neuronală Convoluțională (CNN) folosind TensorFlow și Keras. Structura e clasică și stabilă:

Straturi Conv2D și MaxPooling2D ca să extragă tiparele vizuale (margini, unghiuri) din imagini.

Un strat Flatten care aduce datele pe o singură dimensiune.

Straturi Dense pentru a clasifica imaginile în literele de la A la Y.

Am adăugat și un Dropout(0.5) pentru a dezactiva random neuroni în timpul antrenamentului, chestie care previne overfitting-ul (adică modelul să nu memoreze pozele pe de rost, ci să învețe logica).

După 15 epoci de antrenament, codul salvează toată "experiența" în fișierul model.h5. E un model lightweight, optimizat să ruleze lejer pe un procesor normal.

Flow-ul complet (main.py)
Când pornești camera, MediaPipe izolează mâna, redimensionează pătrățelul la 28x28 pixeli (exact cum a fost antrenat modelul) și îl dă rețelei CNN să ghicească litera. Literele se adună progresiv în cuvinte.

Când arăți semnul păcii, se întâmplă magia de la final: textul brut, care poate mai are greșeli sau îi lipsește punctuația, este aruncat printr-un prompt către API-ul Gemini 2.5 Flash. Gemini corectează instant propoziția, o face să sune uman, și o trimite mai departe spre ElevenLabs. ElevenLabs generează fișierul audio cu o voce naturală și îl redă în difuzoare.

E construit să fie modular, rulând pe un mediu virtual curat setat din requirements.txt, cu caching pe partea de AI ca să nu ardă aiurea credite la request-uri repetate.
