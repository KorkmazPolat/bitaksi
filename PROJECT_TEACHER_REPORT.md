# HR Assistant Chatbot - Detaylı Sistem Raporu

## 1. Projenin Amacı ve Problem Tanımı
Bu proje, kurum içi İnsan Kaynakları (IK) dokümanlarından bilgi çekerek çalışan sorularını yanıtlayan bir **RAG (Retrieval-Augmented Generation)** sistemidir.

Klasik chatbot yaklaşımlarında model, eğitim verisine veya genel dünya bilgisine dayanarak cevap üretir. Bu yaklaşım kurum politikaları gibi güncel ve resmi bilgi gerektiren ortamlarda risklidir. Çünkü:
- Hallusinasyon (olmayan kural üretme) olabilir.
- Kaynak gösterimi zayıf olur.
- Denetlenebilirlik düşer.

Bu proje bu riskleri azaltmak için şu prensibi uygular:
- **Önce dokümandan ilgili kanıtları getir (retrieval), sonra sadece bu kanıtlara dayanarak cevap üret (generation).**

Bu nedenle sistemin temel hedefleri:
1. Kurum içi bilgiye dayalı, güvenilir cevap üretmek
2. Sayfa/bölüm referanslarıyla izlenebilirlik sağlamak
3. Zor sorularda retrieval başarısını artırmak için fallback stratejileri uygulamak

---

## 2. Yüksek Seviyeli Mimari
Sistem 3 ana bloktan oluşur:

1. **Ingestion (Belge işleme ve indexleme)**
2. **Retrieval (Kanıt bulma ve grounding)**
3. **Generation (Cevap üretimi ve kaynak döndürme)**

API katmanı FastAPI ile sağlanır.

Ana endpointler:
- `POST /documents/ingest`: Belge yükleme ve indexleme
- `POST /chat/`: Soru-cevap
- `POST /evaluation/retrieval`, `POST /evaluation/generation`: Kalite ölçümü

Servis içinde temel akış:
- Kullanıcı sorusu -> `SmartGroundingRetriever`
- Elde edilen chunk'lar -> `ResponseGenerator`
- Sonuç -> cevap + kaynaklar + kullanılan strateji + denenen sorgular

---

## 3. Ingestion Pipeline (Belgeyi Bilgi Tabanına Dönüştürme)
Ingestion süreci `DocumentIndexer.ingest_file()` ile yürür ve şu adımlardan oluşur:

### 3.1 Doküman Ayrıştırma (`DocumentProcessor`)
Desteklenen tipler: `.pdf`, `.docx`, `.doc`

- **PDF** için:
  - Her sayfanın metni çıkarılır (`extract_text`).
  - Tablo ham verileri alınır (`extract_tables`).
  - Sayfa görseli PNG olarak render edilir.
  - Görseller `data/pages/{doc_id}/page_{n}.png` altına kaydedilir.
  - Aynı görsel base64 olarak bellekte tutulur (vision extraction için).

- **DOCX** için:
  - Paragraflar alınır, belirli sayıda paragraf bir "sanal sayfa" olarak gruplanır.
  - Böylece PDF'deki sayfa kavramına benzer bir yapı oluşturulur.

Çıktı: `DocumentPage` listesi
- `doc_id`, `page_num`, `text`, `image_b64`, `metadata` vb.

### 3.2 Semantik Chunking (`SemanticChunker`)
Her sayfa metni, hiyerarşik ve semantik kurallarla parçalara ayrılır.

### 3.3 Görsel İçerik Çıkarma (`VisionExtractor`)
Sayfanın görselinden:
- tablo,
- grafik,
- figür,
- görselde kalan ek metin
çıkarılarak metne dönüştürülür.

Bu çıktı ayrıca chunk olarak indexlenir (`content_type = visual`).

### 3.4 Çift Indexe Yazma (`DocumentIndexer`)
Sistem iki ayrı Chroma koleksiyonuna yazar:

1. **Raw index (`hr_docs_raw`)**
- Metin chunk'ları
- Vision chunk'ları
- Metadata: `doc_id`, `source`, `page_num`, `section`, `breadcrumb`, `chunk_index`, `token_count`, `parent_text` vb.

2. **Relatives index (`hr_docs_relatives`)**
- Her chunk için LLM ile üretilen muhtemel kullanıcı soruları + follow-up sorular
- Amaç: Kullanıcının doğal dil sorusunu içerik yerine soru kalıbı üzerinden de yakalamak

Not: Vision extraction ve relatives üretimi `ThreadPoolExecutor` ile paralel çalıştırılır; bu tasarım ingestion süresini düşürür.

---

## 4. Chunking Mekanizması (Detaylı)
Projedeki chunking, klasik "N karakterde kes" yaklaşımı değildir. `HierarchicalSemanticChunker` 5 katmanlı bir pipeline uygular.

## 4.1 Katman 1 - Yapısal Ayrıştırma (Structural Parse)
- Başlıklar regex ile yakalanır:
  - numaralı başlıklar (örn. `1.2`, `2)`),
  - ALL-CAPS başlıklar,
  - markdown başlıklar (`##`).
- Başlık seviyelerine göre bir hiyerarşi çıkarılır.
- Her bölüm için breadcrumb oluşturulur (örn. `Izin Politikasi > Yillik Izin`).

Kazanım:
- Chunk sadece metin parçası değil, dokümandaki bağlamsal konumunu da taşır.

## 4.2 Katman 2 - Cümle Segmentasyonu
- NLTK (`sent_tokenize`, Turkish-aware) ile cümleleme yapılır.
- NLTK yoksa regex fallback kullanılır.
- Chunk sınırlarının cümle ortasından geçmesi engellenir.

Kazanım:
- Embedding kalitesi artar, semantik bozulma azalır.

## 4.3 Katman 3 - Semantik Sınır Tespiti
- Cümle embedding'leri alınır (`SentenceTransformer`).
- Ardışık cümleler arası cosine distance hesaplanır.
- Mesafe dağılımının yüksek yüzdelik dilimindeki (varsayılan %95) noktalar konu geçişi kabul edilir.

Kazanım:
- Bölünme noktaları rastgele değil, anlam kaymasına göre belirlenir.

## 4.4 Katman 4 - Token Bütçesi ile Boyutlandırma
- Token sayımı öncelikli olarak `tiktoken` ile yapılır.
- Küçük parçalar birleştirilir.
- Büyük parçalar token penceresi + overlap ile bölünür.
- Kullanılan ana parametreler:
  - `chunk_max_tokens` (varsayılan 512)
  - `chunk_overlap_tokens` (varsayılan 128)
  - `chunk_min_tokens` (varsayılan 50)

Kazanım:
- Chunk'lar model limitlerine uyumlu olur.
- Boundary etkisi overlap ile azaltılır.

## 4.5 Katman 5 - Context Enrichment
- Chunk metninin başına breadcrumb eklenir (`Bolum: ...`).
- Parent section metni metadata'ya konur (`parent_text`).
- Seviye, token_count gibi sinyaller metadata'ya eklenir.

Kazanım:
- Aynı cümle farklı bölümde geçse bile bağlamı korunur.
- Retrieval ve generation daha doğru referans verir.

---

## 5. Retrieval Sistemi ve Grounding Mantığı
Retrieval katmanının ana sınıfı: `SmartGroundingRetriever`

Amaç:
- Kullanıcı sorusuna karşı ilgili chunk'ları bulmak
- Yetersiz sonuç varsa fallback stratejileriyle recall artırmak
- Grounded olmayan durumda generation katmanına "cevap verme" dedirtmek

### 5.1 Temel Retrieval (`BaseRetriever`)
- Chroma `query()` ile benzerlik araması yapılır.
- `distance` değeri `score = 1 - distance` ile benzerliğe çevrilir.
- `score_threshold` altı sonuçlar elenir.
- Sonuçlar skorla sıralanır.

Raw index ve relatives index için ayrı retriever kullanılır:
- Raw için daha sıkı eşik (`similarity_threshold`)
- Relatives için daha esnek eşik (`relatives_score_threshold`)

### 5.2 Raw + Relatives Birleştirme
`_retrieve_and_augment()` adımında:
1. Raw'dan doğrudan chunk'lar çekilir.
2. Relatives'dan soru benzerliği yakalanır.
3. Relatives hit metadata'sındaki parent `chunk_id` alınır.
4. Parent chunk'lar raw koleksiyondan `fetch_by_ids()` ile çekilir.
5. Tüm sonuçlar deduplicate edilir.

Bu yaklaşım özellikle şu durumda güçlüdür:
- Kullanıcı sorusu, dokümanda geçen birebir cümleden farklıysa
- Ama relatives index içinde soru-formunda temsil edilmişse

### 5.3 Yeterlilik Kontrolü
Retriever, sonucu yeterli saymak için:
- En az `MIN_RESULTS` adet chunk
- Ve bu chunk'ların threshold üstünde olması
koşulunu arar.

Yetersizse fallback zinciri başlatılır.

---

## 6. Fallback Stratejileri (Sıralı Kurtarma)
Sıra ve amaç:

## 6.1 Fallback 1 - Query Expansion
- LLM, sorgunun semantik varyantlarını üretir.
- Örn: "Yıllık izin" -> "senelik izin", "kaç gün ücretli izin" vb.
- Her varyant için retrieval yapılıp sonuçlar birleştirilir.

Ne zaman işe yarar:
- Terminoloji farkı varsa
- Kullanıcı farklı ifade kullandıysa

## 6.2 Fallback 2 - HyDE
- LLM, sorguya yönelik "hipotetik bir politika pasajı" üretir.
- Retrieval bu sentetik metinle yapılır.

Ne zaman işe yarar:
- Soru çok kısa/soyut olduğunda
- Embedding alanında daha iyi bir query temsili gerektiğinde

## 6.3 Fallback 3 - Query Decomposition
- LLM, karmaşık soruyu 2-4 alt soruya böler.
- Her alt soru ayrı retrieve edilir.
- Sonuçlar birleştirilir.

Ne zaman işe yarar:
- Çok parçalı sorular
- Aynı soruda birden fazla IK konusu varsa

Son durumda halen sonuç yoksa:
- `grounded = false`, `strategy_used = none`
- Generation katmanı yanıt üretmeyi reddeder.

---

## 7. Generation Katmanı
Ana sınıf: `ResponseGenerator`

Temel prensip: **RAG-mandatory**

### 7.1 Grounded Olmayan Durum
- Retrieval boşsa veya grounded false ise,
- Sistem sabit bir güvenli mesaj döndürür:
  - ilgili doküman bulunamadığı belirtilir,
  - kullanıcı soruyu yeniden ifade etmeye veya IK ile iletişime yönlendirilir.

### 7.2 Grounded Durum
- Retrieve edilen chunk'lar bağlam bloğu halinde modele verilir.
- Prompt, modeli şu kurallara zorlar:
  - yalnızca verilen bağlamdan yanıtla,
  - kaynak/doküman/sayfa referansı ver,
  - spekülasyon yapma.

### 7.3 Kaynakların Döndürülmesi
Sadece metin cevap değil, kaynak listesi de döner:
- `document`
- `page`
- `section`
- `score`
- `chunk_text` (frontend highlight için)

Böylece sistem "neden bu cevabı verdi" sorusuna denetlenebilir cevap sunar.

---

## 8. API ve Servis Katmanı

### 8.1 `ChatService` Orkestrasyonu
`answer(query, history)` çağrısında:
1. Retriever çalışır
2. Generation çalışır
3. Sonuç tek response modelinde döner

### 8.2 `/chat/` Endpoint Davranışı
- Girdi doğrulaması: boş sorgu ve maksimum uzunluk kontrolü
- Geçmiş mesajlar role filtresiyle alınır (`user`, `assistant`)
- Çıktı:
  - `answer`
  - `sources`
  - `strategy_used`
  - `grounded`
  - `queries_tried`

### 8.3 `/documents/ingest` Endpoint Davranışı
- Dosya uzantısı ve boyut doğrulaması
- Geçici dosyaya yazma
- `DocumentIndexer.ingest_file()` çağrısı
- İstatistik döndürme:
  - sayfa sayısı,
  - text chunk sayısı,
  - visual chunk sayısı,
  - relatives sayısı

---

## 9. Değerlendirme (Evaluation) Tasarımı
Sistemin başarısı iki ayrı boyutta ölçülür.

## 9.1 Retrieval Metrikleri
1. **Recall@K**
- Top-K içinde bulunan ilgili chunk oranı

2. **MRR (Mean Reciprocal Rank)**
- İlk doğru sonucun sırasına duyarlı ölçüm

3. **NDCG@K**
- Sıralamanın kalite etkisini logaritmik indirgemeyle ölçer

## 9.2 Generation Metrikleri
1. **Faithfulness**
- Cevaptaki iddiaların verilen bağlamda desteklenme oranı

2. **Answer Relevancy**
- Soruyla ilgi düzeyi (1-5 puan -> 0-1 normalize)

Bu metrikler hem tek örnek hem batch endpointlerle çalıştırılabilir.

---

## 10. Konfigürasyon ve Ayarlanabilir Parametreler
Sistem `pydantic-settings` ile `.env` üzerinden yönetilir.

Kritik parametre grupları:
- Model seçimi: `llm_provider`, `llm_model`, `vision_model`, `embedding_model`
- Retrieval: `top_k`, `fallback_top_k`, `similarity_threshold`, `relatives_score_threshold`
- Chunking: `chunk_max_tokens`, `chunk_overlap_tokens`, `chunk_min_tokens`, `semantic_split_percentile`
- Ingestion: `batch_upsert_size`, `min_chunk_length_for_relatives`
- API güvenlik/sınır: `max_query_length`, `max_upload_size_mb`

Bu yapı sayesinde sistem farklı doküman tiplerine ve farklı kalite/hedef dengesine göre kalibre edilebilir.

---

## 11. Uçtan Uca Örnek Akış
Soru: "Yıllık izin hakkım kaç gün ve nasıl onaylanıyor?"

1. API sorguyu alır, validate eder.
2. Smart retriever direct arama yapar.
3. Sonuç yetersizse expansion/HyDE/decomposition devreye girer.
4. Raw + relatives sonuçları birleştirilip dedup edilir.
5. Grounded sonuçlar generation'a geçer.
6. Model yalnızca verilen chunk'lardan cevap üretir.
7. Cevap, kaynaklar ve kullanılan strateji ile döndürülür.

---

## 12. Sistem Güçlü Yönleri ve Sınırlılıkları

### Güçlü Yönler
- RAG-mandatory tasarım ile hallusinasyon riski azaltılmıştır.
- Chunking mimarisi semantik ve hiyerarşik açıdan güçlüdür.
- Relatives index + fallback zinciri retrieval recall'unu artırır.
- Sayfa/bölüm referansları denetlenebilirlik sağlar.
- Evaluation katmanı sayesinde iteratif iyileştirme mümkündür.

### Sınırlılıklar
- OCR/parse kalitesi düşük belgelerde performans düşebilir.
- Fallback adımları ek gecikme ve API maliyeti üretir.
- LLM-as-judge metrikleri tamamen objektif değildir.
- Relatives üretiminin kalitesi modele ve prompt davranışına bağlıdır.

---

## 13. Sonuç
Bu proje, basit bir chatbot değil; **belge-merkezli, kanıta dayalı bir IK bilgi sistemi**dir.

Teknik olarak en kritik değer önerisi üç bileşenin birleşimidir:
1. Hiyerarşik semantik chunking,
2. Çift indexli ve fallback destekli retrieval orkestrasyonu,
3. Grounding zorunlu cevap üretimi.

Bu kombinasyon, kurumsal kullanımda gerekli olan güvenilirlik, izlenebilirlik ve kalite kontrolünü sağlar.
