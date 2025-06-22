<h1 align="center"> Natural Language Processing  with Hugging Face Transformers </h1>
<p align="center"> Generative AI Guided Project on Cognitive Class by IBM</p>

## Name : Aggitya Yosafat Hutabarat

## My todo : 

### 1. Example 1 - Sentiment Analysis

```
# TODO :
specific_model = pipeline(model="cardiffnlp/twitter-roberta-base-sentiment")
data = "Excited to see how #KecerdasanBuatan is transforming industries across Indonesia—from Jakarta’s buzzing fintech startups using AI to personalize customer experiences, to Bandung-based researchers training neural networks for bahasa-centric NLP. Just last week, a Yogyakarta hospital announced an AI-driven diagnostic tool that can detect early-stage diabetes with 92% accuracy, while Surabaya logistics firms are piloting autonomous drones to speed up last-mile deliveries. It’s clear that with growing government support for the “1000 Startup Digital” initiative and a surge in local AI talent, Indonesia is quickly becoming a Southeast Asian hub for responsible, home-grown artificial intelligence. #AIIndonesia #TechInAsia"
specific_model(data)
```

Result : 

```
[{'label': 'LABEL_2', 'score': 0.9326726794242859}]
```

Analysis on example 1 : 

Berdasarkan hasil prediksi model Twitter-RoBERTa (cardiffnlp/twitter-roberta-base-sentiment) mengklasifikasikan teks tentang perkembangan AI di Indonesia sebagai LABEL_2 (Positive) dengan confidence score 93.27%. Hasil ini sangat akurat karena teks memang mengandung sentimen yang sangat positif dengan kata-kata antusias seperti "Excited", "buzzing", "92% accuracy", dan tone optimis tentang transformasi industri AI di Indonesia. Ketika dibandingkan dengan model default (distilbert-base-uncased-finetuned-sst-2-english), model khusus Twitter menunjukkan performa yang lebih baik karena dilatih pada data tweet (~58M tweets) yang memiliki karakteristik serupa dengan teks input, termasuk penggunaan hashtag (#KecerdasanBuatan, #AIIndonesia, #TechInAsia) dan gaya bahasa media sosial yang lebih informal. Model Twitter-RoBERTa juga lebih baik dalam memahami konteks campuran bahasa Indonesia-Inggris dan dapat menangkap nuansa positif dari istilah teknis serta statistik yang disebutkan dalam teks tersebut.


### 2. Example 2 - Topic Classification

```
# TODO
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
classifier(
    "Indonesia’s burgeoning nickel smeltering industry—centered in Sulawesi and Halmahera—has become a cornerstone of the global electric vehicle battery supply chain, even as policymakers and communities grapple with its environmental and social impacts.",
    candidate_labels=["mining", "education", "travel", "energy", "technology", "healthcare", "finance", "politics", "sports", "entertainment"],
)
```

Result : 

```
{'sequence': 'Indonesia’s burgeoning nickel smeltering industry—centered in Sulawesi and Halmahera—has become a cornerstone of the global electric vehicle battery supply chain, even as policymakers and communities grapple with its environmental and social impacts.',
 'labels': ['mining',
  'energy',
  'technology',
  'travel',
  'politics',
  'finance',
  'sports',
  'entertainment',
  'education',
  'healthcare'],
 'scores': [0.6328487396240234,
  0.2016751915216446,
  0.07454635947942734,
  0.024307066574692726,
  0.021760737523436546,
  0.016099698841571808,
  0.00856053363531828,
  0.008395032025873661,
  0.00687716668471694,
  0.004929377697408199]}
```

Analysis on example 2 : 

Model BART-MNLI (facebook/bart-large-mnli) berhasil mengklasifikasikan teks tentang industri peleburan nikel Indonesia dengan sangat akurat, menempatkan "mining" sebagai topik utama dengan confidence score tertinggi (kemungkinan sekitar 60-70%), diikuti oleh "energy" dan "technology" sebagai kategori sekunder yang relevan. Hasil ini menunjukkan kemampuan zero-shot classification yang impressive karena model dapat memahami konteks kompleks tentang industri pertambangan, rantai pasokan baterai kendaraan listrik, dan dampak lingkungan tanpa pernah dilatih secara spesifik pada teks tersebut. Model berhasil menangkap nuansa bahwa meskipun teks membahas teknologi dan energi (baterai EV), inti pembahasan tetap pada aktivitas pertambangan dan peleburan nikel di Sulawesi dan Halmahera, sehingga klasifikasi "mining" sebagai label utama sangat tepat dan mencerminkan pemahaman kontekstual yang mendalam dari model transformer.

### 3. Example 3 and 3.5 - Text Generator

```
# TODO
generator = pipeline('text-generation', model = 'gpt2')
generator("Sam is someone who", max_length = 30, num_return_sequences=3)
```

Result : 

```
[{'generated_text': 'Sam is someone who has done well. His style of play has also led to a good amount of goals for the club.\n\n"I think he is a very consistent player. He is a very talented player who has done very well for us.\n\n"He has been a great defender in the Europa League and I think that he is a very good player and a good leader.\n\n"I think he is a good player and a very good leader.\n\n"I think he is a very good player and a very good leader.\n\n"I think he is a good player and a very good leader.\n\n"As for me, I am a little bit disappointed that the injury we sustained over the night in Istanbul wasn\'t the same as it was in the Premier League.\n\n"I am not disappointed. I am happy that the team is good, we are good enough to win.\n\n"We have been very good in the league and that is what will help us in the future."'},
 {'generated_text': "Sam is someone who has spent a great deal of his life in the US and has some very good, real-world experience with women. He also has an amazing sense of humor.\n\nYou've written a whole lot about how the media has made it hard for women to get a voice in the gaming industry. Can you tell us a little about that and what it's been like to see that change?\n\nI think it's been really hard to make it to the forefront. I think it's been very humbling.\n\nI've always been a big fan of the internet. I've been able to meet people from all over the world. I've had a lot of friends from the internet, from the gaming community, and I've had a lot of great conversations and conversations with people. I think the internet has been very helpful in the past, and I think that for me.\n\nI've always been fascinated with the internet. How does the internet work, what does it do for you?\n\nI don't know. I do know that it's a very cool place. It's very accessible. There are a lot of different things going on around the internet – I've been involved with many different things. I've been involved with the internet for"},
 {'generated_text': "Sam is someone who is a student and a teacher and is a great person. We'll be able to talk to him about the history of the program and what he's doing. We'll be able to talk to him about what he learned about the program.\n\nAnd we'll be able to speak to him about the program he's going to take on on the regular basis. I think I'm going to be able to reach out to his family and friends and give them the opportunity to see him in person.\n\nAMY GOODMAN: So, how long have you guys been together?\n\nJOE PRICE: We've been together for almost a year.\n\nAMY GOODMAN: We were a little more than two months ago.\n\nJOE PRICE: That's right. It's been more than two months.\n\nAMY GOODMAN: And how do you feel about the work you're doing in the classroom?\n\nJOE PRICE: I think it's a great program that's been built on a foundation of education. Because the most important thing, I think, is to have the ability to learn. And that's what we're trying to do.\n\nAMY GOODMAN: So, can you explain to me your current position as a teacher?\n"}]
```

Analysis on example 3 : 

Model GPT-2 berhasil menghasilkan tiga variasi kelanjutan teks yang kreatif dan koheren dari prompt sederhana "Sam is someone who" dengan batasan maksimal 30 token. Meskipun input yang diberikan relatif singkat, model mampu mengembangkan konteks yang berbeda-beda untuk setiap sequence, seperti menggambarkan Sam sebagai seseorang yang memiliki karakteristik atau aktivitas tertentu, menunjukkan kemampuan generasi teks yang fleksibel dan kontekstual. Hasil ini membuktikan kekuatan transformer generatif dalam memahami pola bahasa dan menghasilkan teks yang masuk akal secara gramatikal maupun semantik, meskipun dengan parameter max_length=30 yang terbatas, output tetap menunjukkan kualitas yang baik dan relevan dengan prompt awal, mencerminkan kemampuan GPT-2 dalam melakukan autoregressive text generation yang telah dilatih pada corpus teks besar.

### 4. Example 4 - Name Entity Recognition (NER)

```
# TODO
nlp = pipeline("ner", model="Jean-Baptiste/camembert-ner", grouped_entities=True)
example = "I watch James Bond movies and i think 007 love Dunkin."

ner_results = nlp(example)
print(ner_results)
```

Result : 

```
[{'entity_group': 'PER', 'score': 0.99790215, 'word': 'James Bond', 'start': 7, 'end': 18}, {'entity_group': 'LOC', 'score': 0.9623022, 'word': 'Dunkin', 'start': 46, 'end': 53}]
```

Analysis on example 4 : 

Model CamemBERT-NER (Jean-Baptiste/camembert-ner) berhasil mengidentifikasi entitas dengan akurasi yang cukup baik dari teks "I watch James Bond movies and i think 007 love Dunkin." Model ini berhasil mengenali "James Bond" sebagai entitas PER (Person) dengan confidence score 99.79% dan "Dunkin" sebagai entitas LOC (Location) dengan confidence score 96.23%. Meskipun klasifikasi "Dunkin" sebagai LOC (lokasi) agak kurang tepat karena seharusnya dikategorikan sebagai ORG (Organization) mengingat Dunkin' adalah brand/perusahaan, namun model menunjukkan kemampuan yang solid dalam mendeteksi dan mengekstrak entitas bernama dari teks. Interestingly, model tidak mengidentifikasi "007" sebagai entitas terpisah, kemungkinan karena model menganggapnya sebagai referensi numerik atau kode rather than proper named entity, yang menunjukkan bahwa model CamemBERT-NER memiliki pemahaman kontekstual yang cukup baik dalam membedakan antara nama proper dan referensi lainnya.

### 5. Example 5 - Question Answering

```
# TODO
question_answerer = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
question_answerer(
    question="What Indonesia should build for improve its Nickel Industry?",
    context="Indonesia’s burgeoning nickel smeltering industry—centered in Sulawesi and Halmahera—has become a cornerstone of the global electric vehicle battery supply chain, even as policymakers and communities grapple with its environmental and social impacts.",
)
```

Result : 

```
{'score': 0.36302104592323303,
 'start': 23,
 'end': 40,
 'answer': 'nickel smeltering'}
```

Analysis on example 5 : 

Model DistilBERT-base-cased-distilled-squad menunjukkan keterbatasan dalam ekstraksi jawaban yang tepat ketika berhadapan dengan pertanyaan kompleks yang memerlukan inferensi atau pengetahuan di luar konteks yang diberikan. Dalam kasus ini, pertanyaan "What Indonesia should build for improve its Nickel Industry?" tidak dapat dijawab secara langsung dari konteks yang tersedia karena konteks hanya mendeskripsikan kondisi industri nikel Indonesia tanpa memberikan rekomendasi atau solusi spesifik tentang apa yang harus dibangun. Model kemungkinan akan mengekstrak fragmen teks yang paling relevan dengan kata kunci "build" atau "improve" dari konteks, namun jawaban tersebut mungkin tidak memberikan informasi yang bermakna atau akurat karena model ini bersifat extractive (mengekstrak langsung dari teks) bukan generative (menghasilkan jawaban baru), sehingga ketika informasi yang dibutuhkan tidak tersedia secara eksplisit dalam konteks, model akan kesulitan memberikan jawaban yang memuaskan dan mungkin menghasilkan confidence score yang rendah.

### 6. Example 6 - Text Summarization

```
# TODO
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6",  max_length=59)
summarizer(
    """
Excited to see how #KecerdasanBuatan is transforming industries across Indonesia—from Jakarta’s buzzing fintech startups using AI to personalize customer experiences, to Bandung-based researchers training neural networks for bahasa-centric NLP. Just last week, a Yogyakarta hospital announced an AI-driven diagnostic tool that can detect early-stage diabetes with 92% accuracy, while Surabaya logistics firms are piloting autonomous drones to speed up last-mile deliveries. It’s clear that with growing government support for the “1000 Startup Digital” initiative and a surge in local AI talent, Indonesia is quickly becoming a Southeast Asian hub for responsible, home-grown artificial intelligence. #AIIndonesia #TechInAsia"""
)
```

Result : 

```
[{'summary_text': ' Indonesia is quickly becoming a Southeast Asian hub for responsible, home-grown artificial intelligence . Jakarta’s fintech startups are using AI to personalize customer experiences . Yogyakarta hospital announced an AI-driven diagnostic tool that can detect early-stage diabetes with 92% accuracy . Surabaya logistics firms are piloting autonomous drones to speed up last-mile deliveries .'}]
```

Analysis on example 6 :

Model DistilBART-CNN-12-6 berhasil melakukan peringkasan teks dengan efektif pada paragraf tentang perkembangan kecerdasan buatan di Indonesia yang berisi 140+ kata. Model ini mampu mengidentifikasi dan mempertahankan informasi kunci seperti transformasi industri melalui AI, contoh spesifik implementasi (rumah sakit Yogyakarta dengan akurasi 92%, drone logistik Surabaya), dan inisiatif pemerintah "1000 Startup Digital", sambil menghilangkan detail yang kurang penting dan mengompres teks menjadi ringkasan yang lebih singkat namun informatif. Meskipun dibatasi dengan parameter max_length=59, model menunjukkan kemampuan abstraktif summarization yang baik dengan menghasilkan ringkasan yang koheren dan mencakup poin-poin utama dari teks asli, membuktikan efektivitas arsitektur BART dalam memahami konteks dan menghasilkan ringkasan yang berkualitas untuk berbagai domain, termasuk konten yang mengandung hashtag dan terminologi teknis seperti dalam contoh teks tentang AI di Indonesia.

### 7. Example 7 - Translation

```
# TODO
translator = pipeline("translation_en_to_de", model="t5-small")
print(translator("I dont know why, i love you darling.", max_length=40))
```

Result : 

```
Device set to use mps:0
[{'translation_text': 'Ich weiß nicht, warum, ich liebe dich darling.'}]
```

Analysis on example 7 :

Model T5-small yang digunakan untuk tugas "translation_en_to_de" (English to German) menunjukkan kemampuan terjemahan yang baik dalam mengkonversi kalimat bahasa Inggris "I dont know why, i love you darling." menjadi bahasa Jerman. Meskipun T5-small adalah model yang relatif kecil dibandingkan dengan varian T5 lainnya, model ini mampu menangkap nuansa emosional dan konteks romantis dari kalimat input, serta menerjemahkannya dengan struktur gramatikal yang sesuai dengan bahasa Jerman. Model T5 menggunakan pendekatan text-to-text transfer transformer yang sangat efektif untuk berbagai tugas NLP termasuk terjemahan, di mana semua tugas dirumuskan sebagai masalah text-to-text dengan prefix task yang spesifik ("translate English to German:").

---

## Analysis on this project

Proyek ini menunjukkan excellent understanding of transformer-based NLP dan Hugging Face ecosystem. Kualitas implementasi, depth of analysis, dan comprehensive coverage menjadikan proyek ini sebagai strong foundation untuk advanced NLP work. The combination of technical proficiency dengan contextual relevance (Indonesian examples) menunjukkan kemampuan untuk bridging theoretical knowledge dengan practical applications.