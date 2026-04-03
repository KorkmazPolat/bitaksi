# 01. Sistem Genel Bakış

## 1. Amaç
Bu proje, kurumsal dokümanlardan beslenen ve cevaplarını kaynaklarla ilişkilendiren bir RAG bilgi asistanıdır. Temel hedef, çalışan veya kullanıcı sorularına yalnızca model hafızasıyla değil, doküman kanıtlarıyla cevap vermektir.

Sistem, özellikle mevzuat, şirket rehberi, İK politikası veya prosedür gibi doğruluk gerektiren bilgi alanlarında şu üç problemi çözmeyi hedefler:

- serbest üretimden kaynaklanan halüsinasyon
- kaynaksız cevaplara duyulan güvensizlik
- kullanıcı dili ile doküman dili arasındaki ifade farkı

## 2. Sistem Sınırları
Sistemin mevcut sürümü şu sorumlulukları üstlenmektedir:

- PDF ve DOCX belgelerini ingest etmek
- metin ve görsel içerikleri ayrıştırmak
- semantik ve bağlamsal chunk üretmek
- chunk’ları vektör index’lere yazmak
- kullanıcı sorularında retrieval ve grounding yapmak
- yalnızca bulunan bağlam üzerinden cevap üretmek
- kaynakları kullanıcıya göstermek

Kapsam dışında kalan başlıca alanlar:

- kimlik doğrulama
- yetkilendirme ve rol bazlı erişim
- belge versiyonlama
- üretim seviyesi monitoring/dashboard
- insan onaylı workflow

## 3. Ana Mimari
Sistem beş ana katman halinde düşünülebilir:

1. Doküman alma ve normalizasyon
2. Chunk üretimi ve zenginleştirme
3. Vektör indexleme
4. Retrieval ve grounding
5. Generation, API ve UI

## 4. Temel Tasarım İlkeleri

### 4.1 RAG Zorunluluğu
Sistem, retrieval başarısızsa yine de cevap üretmeye çalışan "best effort chatbot" yaklaşımını tercih etmez. Bunun yerine önce yeterli kanıt aranır, ardından generation başlatılır. Bu ayrım mimari olarak kritiktir.

### 4.2 Explainability
Her cevapla birlikte:

- kullanılan strateji
- denenen sorgular
- kaynak listesi

döndürülür. Böylece sistemin neden belli bir cevaba gittiği anlaşılabilir hale gelir.

### 4.3 Graceful Degradation
Vision extraction, relatives generation veya bazı LLM yardımcı görevleri başarısız olsa bile sistemin tamamının durmaması hedeflenmiştir. Kritik amaç, sistemin tam çökme yerine kademeli değer kaybı yaşamasıdır.

### 4.4 Modülerlik
Ingestion, retrieval, generation ve evaluation katmanları ayrı modüllere bölünmüştür. Bu sayede:

- test etmek kolaylaşır
- bileşen bazlı geliştirme yapılabilir
- gelecekte başka embedding modeli veya vector store ile adaptasyon kolaylaşır

## 5. Veri Akışı
Uçtan uca akış aşağıdaki gibidir:

1. Belge yüklenir.
2. Belge sayfa bazlı işlenir.
3. Metin, tablo ve sayfa görseli çıkarılır.
4. Sayfalar belge-seviye bölümlere dönüştürülür.
5. Semantik olarak tutarlı chunk’lar üretilir.
6. Chunk’lar için gerekirse contextual enrichment eklenir.
7. Raw ve relatives koleksiyonlarına yazım yapılır.
8. Kullanıcı soru sorar.
9. Retriever önce raw, sonra relatives, gerekirse fallback zinciri ile arama yapar.
10. Sonuçlar rerank edilir.
11. Yeterli bağlam varsa cevap üretilir, yoksa güvenli red mesajı döner.


## 6. Sistemdeki En Önemli Teknik Kararlar

### 6.1 Çift Index Yaklaşımı
Sadece chunk embedding araması yapmak yerine ayrıca chunk bazlı üretilmiş soru/follow-up kayıtları da indexlenir. Bu, kullanıcı sorusu ile doküman cümlesi arasında doğrudan yüzeysel benzerlik olmadığında recall’u artırır.

### 6.2 Cross-Page Section Chunking
Chunking artık sadece tek sayfa sınırına bağlı değildir. Bir bölüm ardışık sayfalarda devam ediyorsa aynı mantıksal bölüm içinde ele alınabilir. Bu, özellikle rehber ve mevzuat tipi belgelerde daha doğru chunk sınırları üretir.

### 6.3 Lexical Re-ranking
Embedding skoru tek başına yeterli olmadığından, retrieval sonrası sorgu ile chunk arasındaki kelime örtüşmesi de skora küçük bir katkı yapar. Böylece semantik olarak yakın ama pratikte daha az ilgili chunk’ların üste çıkma ihtimali azalır.

## 7. Kritik Riskler

- Vision JSON çıktıları model sağlayıcısına göre bozulabilir.
- Relatives generation maliyeti büyük belgelerde yükselir.
- Tek doküman yerine çok dokümanlı veri havuzunda source ambiguity artabilir.
- Çok dilli kullanımda embedding performansı tutarsızlaşabilir.
- Belge versiyonlama yoksa eski politika kayıtları kullanıcıya dönebilir.

## 8. Sonuç
Bu sistem bir chatbot arayüzüne sahip olsa da teknik olarak bir belge merkezli bilgi erişim, grounding ve kontrollü cevap üretim altyapısıdır. Projenin asıl gücü, generation kalitesinden önce retrieval ve bağlam disiplinini ciddiye almasıdır.
