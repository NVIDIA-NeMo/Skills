# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import namedtuple

from datasets import load_dataset

MCQFormat = namedtuple("MCQFormat", ["task", "answer_prefix", "placeholder"])


# Hugging Face dataset schema.
class Schema:
    ANSWER: str = "answer"
    LANGUAGE: str = "language"
    DOMAIN: str = "domain"
    QUESTION: str = "question"
    SUBJECT: str = "subject"
    COUNTRY: str = "country"
    REGIONAL_FEATURE: str = "regional_feature"
    LEVEL: str = "level"
    OPTIONS: list[str] = [
        "option_a",
        "option_b",
        "option_c",
        "option_d",
    ]


def load_include_datasets(languages, split):
    return [
        load_dataset("CohereLabs/include-base-44", lang)[split] for lang in languages
    ]

# MMMLU-style greedy fallback: last answer letter anywhere in the response.
LETTER_REGEX = r"\b\(?\s*([A-D]|[أ-د]|[অ]|[ব]|[ড]|[ঢ]|[Ａ]|[Ｂ]|[Ｃ]|[Ｄ])\s*\)?\.?\b"
GREEDY_REGEX = r"[\s\S]*" + LETTER_REGEX

MCQ_FORMATS = {
    "Russian": MCQFormat(
        answer_prefix="Ответ: Давайте подумаем шаг за шагом.",
        task='Ниже приведены вопросы с несколькими вариантами ответов (с ответами) по теме {subject}. Пожалуйста, размышляйте шаг за шагом, а затем завершите свой ответ с "{ans_suffix}", где X - это буква правильного варианта.',
        placeholder="Ответ - ({})",
    ),
    "French": MCQFormat(
        answer_prefix="Réponse: Réfléchissons étape par étape.",
        task='Les questions suivantes sont à choix multiples (avec réponses) sur {subject}. Réfléchissez étape par étape, puis terminez votre réponse par "{ans_suffix}" où X est la lettre correspondant au bon choix.',
        placeholder="La réponse est ({})",
    ),
    "Spanish": MCQFormat(
        answer_prefix="Respuesta: Pensemos paso a paso.",
        task='Las siguientes son preguntas de opción múltiple (con respuestas) sobre {subject}. Piense paso a paso y luego termine su respuesta con "{ans_suffix}" donde X es la letra de la opción correcta.',
        placeholder="La respuesta es ({})",
    ),
    "German": MCQFormat(
        answer_prefix="Antwort: Denken wir Schritt für Schritt nach.",
        task='Nachfolgend finden Sie Multiple-Choice-Fragen (mit Antworten) zu {subject}. Denken Sie Schritt für Schritt nach und beenden Sie Ihre Antwort mit "{ans_suffix}", wobei X der richtige Buchstabe ist.',
        placeholder="Die Antwort ist ({})",
    ),
    "Portuguese": MCQFormat(
        answer_prefix="Resposta: Vamos pensar passo a passo.",
        task='A seguir estão perguntas de múltipla escolha (com respostas) sobre {subject}. Pense passo a passo e termine sua resposta com "{ans_suffix}" onde X é a letra da opção correta.',
        placeholder="A resposta é ({})",
    ),
    "Italian": MCQFormat(
        answer_prefix="Risposta: Ragioniamo passo dopo passo.",
        task='Di seguito sono riportate domande a scelta multipla (con risposte) su {subject}. Si prega di ragionare passo dopo passo e terminare la risposta con "{ans_suffix}", dove X è la lettera dell\'opzione corretta.',
        placeholder="La risposta è ({})",
    ),
    "Serbian": MCQFormat(
        answer_prefix="Odgovor: Razmislimo korak po korak.",
        task='Следе питања са вишеструким избором (са одговорима) о {subject}. Molimo vas da razmislite korak po korak i završite svoj odgovor sa "{ans_suffix}", gde je X slovo tačne opcije.',
        placeholder="Odgovor je ({})",
    ),
    "Ukrainian": MCQFormat(
        answer_prefix="Відповідь: Давайте подумаємо крок за кроком.",
        task='Нижче наведено запитання з кількома варіантами відповідей (з відповідями) про {subject}. Будь ласка, подумайте крок за кроком і закінчіть свою відповідь "{ans_suffix}", де X – літера правильного варіанту.',
        placeholder="Відповідь: ({})",
    ),
    "Hungarian": MCQFormat(
        answer_prefix="Válasz: Gondolkodjunk lépésről lépésre.",
        task='Az alábbiak feleletválasztós kérdések (válaszokkal) a következővel kapcsolatban: {subject}. Kérjük, gondolkodjon lépésről lépésre, és a válaszát a(z) "{ans_suffix}" kifejezéssel fejezze be, ahol X a helyes válasz betűjele.',
        placeholder="A válasz ({})",
    ),
    "Vietnamese": MCQFormat(
        answer_prefix="Trả lời: Hãy suy nghĩ từng bước một.",
        task='Sau đây là các câu hỏi trắc nghiệm (có đáp án) về {subject}. Vui lòng suy nghĩ từng bước, sau đó kết thúc câu trả lời của bạn bằng "{ans_suffix}", trong đó X là chữ cái của lựa chọn đúng.',
        placeholder="Câu trả lời là ({})",
    ),
    "Urdu": MCQFormat(
        answer_prefix="جواب: آئیے قدم بہ قدم سوچتے ہیں۔",
        task='ذیل میں {subject} کے بارے میں متعدد انتخابی سوالات (جوابات کے ساتھ) ہیں۔ براہ کرم قدم بہ قدم سوچیں، اور پھر اپنے جواب کو "{ans_suffix}" کے ساتھ ختم کریں، جہاں X درست آپشن کا حرف ہے۔',
        placeholder="جواب ({}) ہے",
    ),
    "Indonesian": MCQFormat(
        answer_prefix="Jawaban: Mari berpikir langkah demi langkah.",
        task='Berikut ini adalah pertanyaan pilihan ganda (dengan jawaban) tentang {subject}. Harap berpikir langkah demi langkah, lalu akhiri jawaban Anda dengan "{ans_suffix}", di mana X adalah huruf pilihan yang benar.',
        placeholder="Jawabannya adalah ({})",
    ),
    "Korean": MCQFormat(
        answer_prefix="답변: 한 단계씩 생각해 봅시다.",
        task='다음은 {subject}에 대한 객관식 질문(답변 포함)입니다. 단계적으로 생각한 다음 "{ans_suffix}"로 답변을 마무리하세요. 여기서 X는 올바른 선택지 문자입니다.',
        placeholder="답은 ({})입니다",
    ),
    "Hindi": MCQFormat(
        answer_prefix="उत्तर: चलिए चरण-दर-चरण सोचते हैं।",
        task='निम्नलिखित {subject} के बारे में बहुविकल्पीय प्रश्न (उत्तर सहित) हैं। चरण-दर-चरण सोचें और फिर अपने उत्तर को "{ans_suffix}" के साथ समाप्त करें जहां X सही विकल्प का अक्षर है।',
        placeholder="उत्तर है ({})",
    ),
    "Arabic": MCQFormat(
        answer_prefix="الإجابة: دعنا نفكر خطوة بخطوة.",
        task='فيما يلي أسئلة اختيارية متعددة (مع الإجابات) حول {subject}. فكر خطوة بخطوة ثم أنهِ إجابتك بـ "{ans_suffix}" حيث X هو حرف الخيار الصحيح.',
        placeholder="الإجابة هي ({})",
    ),
    "Telugu": MCQFormat(
        answer_prefix="సమాధానం: దశలవారీగా ఆలోచిద్దాం.",
        task='{subject} గురించిన బహుళ-ఎంపిక ప్రశ్నలు (సమాధానాలతో) క్రిందివి. దయచేసి దశలవారీగా ఆలోచించి, మీ సమాధానాన్ని "{ans_suffix}"తో ముగించండి, ఇక్కడ X సరైన ఎంపిక అక్షరం.',
        placeholder="సమాధానం ({})",
    ),
    "Bengali": MCQFormat(
        answer_prefix="উত্তর: আসুন ধাপে ধাপে চিন্তা করি।",
        task='নিম্নলিখিতগুলি {subject} সম্পর্কে বহু-পছন্দের প্রশ্ন (উত্তর সহ)। ধাপে ধাপে চিন্তা করুন এবং তারপর আপনার উত্তর "{ans_suffix}" দিয়ে শেষ করুন যেখানে X হল সঠিক বিকল্পের অক্ষর।',
        placeholder="উত্তর হল ({})",
    ),
    "Nepali": MCQFormat(
        answer_prefix="उत्तर: चरणबद्ध रूपमा सोचौं।",
        task='निम्न {subject} को बारेमा बहु-छनौट प्रश्नहरू (उत्तरहरू सहित) छन्। कृपया चरणबद्ध रूपमा सोच्नुहोस् र आफ्नो उत्तर "{ans_suffix}" बाट अन्त्य गर्नुहोस्, जहाँ X सही विकल्पको अक्षर हो।',
        placeholder="उत्तर ({}) हो।",
    ),
    "Japanese": MCQFormat(
        answer_prefix="回答：一歩一歩考えていきましょう。",
        task="以下は、{subject} に関する複数選択の質問（回答付き）です。段階的に考え、最後に「{ans_suffix}」と回答を締めくくってください。Xは正解の選択肢を示す文字です。",
        placeholder="答えは ({}) です",
    ),
    "Chinese": MCQFormat(
        answer_prefix="答案：让我们一步一步地思考。",
        task='以下是关于 {subject} 的多项选择题（附答案）。请逐步思考，然后以"{ans_suffix}"结束您的回答，其中X是正确的选项字母。',
        placeholder="答案是 ({})",
    ),
    "Albanian": MCQFormat(
        answer_prefix="Përgjigje: Le ta mendojmë hap pas hapi.",
        task='Më poshtë janë pyetjet me zgjedhje të shumëfishta (me përgjigje) rreth {subject}. Mendo hap pas hapi dhe pastaj përfundoje përgjigjen tënde me "{ans_suffix}" ku X është varianti i saktë.',
        placeholder="përgjigjja është ({})",
    ),
    "Armenian": MCQFormat(
        answer_prefix="Պատասխան․ Եկեք մտածենք քայլ առ քայլ։",
        task='Ստորև բերված են բազմակի ընտրության հարցեր (պատասխաններով) {subject}-ի վերաբերյալ: Մտածեք քայլ առ քայլ, ապա ավարտեք ձեր պատասխանը "{ans_suffix}"-ով, որտեղ X-ը ճիշտ տարբերակի տառն է։',
        placeholder="պատասխանը՝ ({})",
    ),
    "Azerbaijani": MCQFormat(
        answer_prefix="Cavab: Gəlin addım-addım düşünək.",
        task='Aşağıdakılar {subject} haqqında çoxseçimli suallardır (cavabları ilə). Addım-addım düşünün və cavabınızı "{ans_suffix}" ilə bitirin, burada X düzgün variantın hərfidir.',
        placeholder="cavab ({})",
    ),
    "Basque": MCQFormat(
        answer_prefix="Erantzuna: Pentsa dezagun pausoz pauso.",
        task='Honako hauek aukera anitzeko galderak dira (erantzunekin) {subject}-i buruz. Pentsatu pausoz pauso eta amaitu zure erantzuna "{ans_suffix}"-ekin, non X aukera zuzenaren letra den.',
        placeholder="erantzuna ({})",
    ),
    "Belarusian": MCQFormat(
        answer_prefix="Адказ: Давайце падумаем крок за крокам.",
        task='Ніжэй прыведзены пытанні з некалькімі варыянтамі адказаў (з адказамі) пра {subject}. Падумайце крок за крокам і завершце адказ фразай "{ans_suffix}", дзе X — літара правільнага варыянта.',
        placeholder="адказ ({})",
    ),
    "Bulgarian": MCQFormat(
        answer_prefix="Отговор: Нека мислим стъпка по стъпка.",
        task='Следват въпроси с избираем отговор (с отговори) за {subject}. Мислете стъпка по стъпка и завършете отговора си с "{ans_suffix}", където X е буквата на правилния избор.',
        placeholder="отговорът е ({})",
    ),
    "Croatian": MCQFormat(
        answer_prefix="Odgovor: Razmislimo korak po korak.",
        task='Slijede pitanja s višestrukim izborom (s odgovorima) o {subject}. Razmišljajte korak po korak i završite svoj odgovor s "{ans_suffix}", gdje je X slovo točnog odgovora.',
        placeholder="odgovor je ({})",
    ),
    "Dutch": MCQFormat(
        answer_prefix="Antwoord: Laten we stap voor stap nadenken.",
        task='Hieronder staan ​​meerkeuzevragen (met antwoorden) over {subject}. Denk stap voor stap en eindig uw antwoord met "{ans_suffix}", waarbij X de letter van het juiste antwoord is.',
        placeholder="het antwoord is ({})",
    ),
    "Estonian": MCQFormat(
        answer_prefix="Vastus: Mõtleme samm-sammult.",
        task='Järgmised on valikvastustega küsimused (koos vastustega) teemal {subject}. Mõelge samm-sammult ja lõpetage oma vastus sõnadega "{ans_suffix}", kus X on õige vastuse täht.',
        placeholder="vastus on ({})",
    ),
    "Finnish": MCQFormat(
        answer_prefix="Vastaus: Mietitään askel askeleelta.",
        task='Seuraavat ovat monivalintakysymyksiä (vastauksineen) aiheesta {subject}. Mieti askel askeleelta ja päätä vastauksesi "{ans_suffix}"-merkintään, jossa X on oikea vastausvaihtoehto.',
        placeholder="vastaus on ({})",
    ),
    "Georgian": MCQFormat(
        answer_prefix="პასუხი: ვიფიქროთ ეტაპობრივად.",
        task='ქვემოთ მოცემულია რამდენიმე არჩევანის კითხვები (პასუხებით) {subject}-ის შესახებ. იფიქრეთ ეტაპობრივად და დაასრულეთ თქვენი პასუხი "{ans_suffix}"-ით, სადაც X სწორ ვარიანტს აღნიშნავს.',
        placeholder="პასუხია ({})",
    ),
    "Greek": MCQFormat(
        answer_prefix="Απάντηση: Ας σκεφτούμε βήμα-βήμα.",
        task='Ακολουθούν ερωτήσεις πολλαπλής επιλογής (με απαντήσεις) σχετικά με το {subject}. Σκεφτείτε βήμα-βήμα και ολοκληρώστε την απάντησή σας με "{ans_suffix}", όπου X είναι το σωστό γράμμα.',
        placeholder="η απάντηση είναι ({})",
    ),
    "Hebrew": MCQFormat(
        answer_prefix="תשובה: בואו נחשוב צעד אחר צעד.",
        task='להלן שאלות ברירות רבות (עם תשובות) על {subject}. חשבו צעד אחר צעד וסיימו את תשובתכם עם "{ans_suffix}", כאשר X הוא האות של התשובה הנכונה.',
        placeholder="התשובה היא ({})",
    ),
    "Kazakh": MCQFormat(
        answer_prefix="Жауап: Қадам-қадам ойланайық.",
        task='Төменде {subject} туралы бірнеше таңдаулы сұрақтар (жауаптары бар) берілген. Қадамдап ойланып, жауабыңызды "{ans_suffix}" деп аяқтаңыз, мұнда X — дұрыс жауаптың әрпі.',
        placeholder="жауабы ({})",
    ),
    "Lithuanian": MCQFormat(
        answer_prefix="Atsakymas: Pagalvokime žingsnis po žingsnio.",
        task='Toliau pateikiami klausimai su atsakymų variantais (su atsakymais) apie {subject}. Mąstykite žingsnis po žingsnio ir užbaikite atsakymą fraze "{ans_suffix}", kur X yra teisingo varianto raidė.',
        placeholder="atsakymas ({})",
    ),
    "Malay": MCQFormat(
        answer_prefix="Jawapan: Mari kita fikir selangkah demi selangkah.",
        task='Berikut ialah soalan aneka pilihan (dengan jawapan) tentang {subject}. Fikir secara berperingkat dan tamatkan jawapan anda dengan "{ans_suffix}", di mana X ialah huruf pilihan yang betul.',
        placeholder="jawapannya ialah ({})",
    ),
    "Malayalam": MCQFormat(
        answer_prefix="ഉത്തരം: നമുക്ക് ഘട്ടംഘട്ടമായി ചിന്തിക്കാം.",
        task='ഇനിപ്പറയുന്നവ {subject} നെക്കുറിച്ചുള്ള മൾട്ടിപ്പിൾ ചോയ്‌സ് ചോദ്യങ്ങളാണ് (ഉത്തരങ്ങളോടെ). ഘട്ടംഘട്ടമായി ചിന്തിച്ച് നിങ്ങളുടെ ഉത്തരത്തിന്റെ അവസാനം "{ans_suffix}" എന്ന് ചേർക്കുക, ഇവിടെ X ശരിയായ ഓപ്ഷന്റെ അക്ഷരമാണ്.',
        placeholder="ഉത്തരം ({})",
    ),
    "North Macedonian": MCQFormat(
        answer_prefix="Одговор: Да размислуваме чекор по чекор.",
        task='Следниве се прашања со повеќекратен избор (со одговори) за {subject}. Размислувајте чекор по чекор и завршете го одговорот со "{ans_suffix}", каде X е буквата на точниот избор.',
        placeholder="одговорот е ({})",
    ),
    "Persian": MCQFormat(
        answer_prefix="پاسخ: بیایید گام به گام فکر کنیم.",
        task='در زیر سؤالات چند گزینه‌ای (همراه با پاسخ) در مورد {subject} آمده است. لطفاً مرحله‌به‌مرحله فکر کنید و سپس پاسخ خود را با "{ans_suffix}" به پایان برسانید، که در آن X حرف گزینهٔ درست است.',
        placeholder="پاسخ ({})",
    ),
    "Polish": MCQFormat(
        answer_prefix="Odpowiedź: Zastanówmy się krok po kroku.",
        task='Poniżej znajdują się pytania wielokrotnego wyboru (z odpowiedziami) na temat {subject}. Myśl krok po kroku i zakończ odpowiedź frazą "{ans_suffix}", gdzie X oznacza poprawną literę odpowiedzi.',
        placeholder="odpowiedź to ({})",
    ),
    "Tagalog": MCQFormat(
        answer_prefix="Sagot: Mag-isip tayo nang paisa-isang hakbang.",
        task='Ang mga sumusunod ay maramihang pagpipiliang tanong (na may mga sagot) tungkol sa {subject}. Mag-isip nang sunod-sunod at tapusin ang iyong sagot sa "{ans_suffix}", kung saan ang X ang tamang titik.',
        placeholder="ang sagot ay ({})",
    ),
    "Tamil": MCQFormat(
        answer_prefix="பதில்: படிப்படியாக யோசிப்போம்.",
        task='பின்வருபவை {subject} பற்றிய பல தேர்வு கேள்விகள் (பதில்களுடன்) உள்ளன. படிப்படியாக யோசித்து, உங்கள் பதிலை "{ans_suffix}" என முடிக்கவும், இதில் X சரியான தேர்வின் எழுத்தாகும்.',
        placeholder="பதில் ({})",
    ),
    "Turkish": MCQFormat(
        answer_prefix="Cevap: Adım adım düşünelim.",
        task='Aşağıda {subject} ile ilgili çoktan seçmeli sorular (cevaplarıyla birlikte) bulunmaktadır. Adım adım düşünün ve cevabınızı "{ans_suffix}" ifadesiyle bitirin; burada X doğru seçeneğin harfidir.',
        placeholder="cevap ({})",
    ),
    "Uzbek": MCQFormat(
        answer_prefix="Javob: Keling, bosqichma-bosqich o'ylaylik.",
        task='Quyida {subject} boʻyicha koʻp tanlovli savollar (javoblari bilan) keltirilgan. Bosqichma-bosqich o‘ylang va javobingizni "{ans_suffix}" bilan yakunlang, bu yerda X to‘g‘ri variant harfidir.',
        placeholder="javob ({})",
    ),
}

SUPPORTED_LANGUAGES = sorted(MCQ_FORMATS.keys())


def digit_to_letter(digit):
    return chr(ord("A") + int(digit))


def build_extract_regex(language):
    # Phrase-anchored regex from the language's placeholder, then greedy fallback.
    placeholder = MCQ_FORMATS[language].placeholder
    phrase_regex = placeholder.replace("({})", r"\(?([ABCD])\)?\s*")
    return [phrase_regex, GREEDY_REGEX]


def normalize_entry_field(entry, key):
    return (entry.get(key, "") or "").replace(" ", "_")


def copy_other_fields(entry):
    return {
        k: normalize_entry_field(entry, k)
        for k in [
            Schema.COUNTRY,
            Schema.REGIONAL_FEATURE,
            Schema.LEVEL,
            Schema.SUBJECT,
            Schema.LANGUAGE,
            Schema.DOMAIN,
        ]
    }


def build_prompt(question, options_text, language, subject):
    fmt = MCQ_FORMATS[language]
    task = fmt.task.format(subject=subject, ans_suffix=fmt.placeholder.format("X"))
    return f"{task}\n\n{question}\n{options_text}\n{fmt.answer_prefix}".strip()


def get_mcq_fields(question, options, language, subject):
    options_dict = {digit_to_letter(i): opt for i, opt in enumerate(options)}
    options_text = "\n".join(f"{k}. {v}" for k, v in options_dict.items())
    prompt = build_prompt(question, options_text, language, subject)
    return {"question": prompt, "options": options_text, **options_dict}
