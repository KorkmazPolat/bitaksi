# 02. Ingestion ve Indexing 

## 1. Amaç
Bu rapor, belge yükleme anından raw/relatives index’lerin oluşmasına kadar olan veri hattını ayrıntılı biçimde anlatır. Projedeki retrieval başarısının büyük bölümü ingestion kalitesine bağlı olduğu için bu katman en kritik mühendislik alanlarından biridir.

## 2. Ingestion Katmanı Bileşenleri

- `DocumentProcessor`
- `VisionExtractor`
- `HierarchicalSemanticChunker`
- `DocumentIndexer`

Bu bileşenlerin ilişkisi:

## 3. DocumentProcessor

### 3.1 PDF İşleme
PDF için her sayfa üzerinde üç ayrı veri tipi elde edilir:

- düz metin
- tablo ham verisi
- görsel render

Bu kararın gerekçesi, kurumsal belgelerde bilgi yalnızca düz paragraf metni olarak bulunmamasıdır. Bir prosedür tablosu veya karar matrisi çoğu zaman metin parser tarafından eksik okunur.

### 3.2 DOCX İşleme
DOCX için gerçek sayfa kavramı garantili olmadığından sanal sayfalama yapılır. Bu, UI ve retrieval tarafında ortak veri modelini korumak için tercih edilir.

### 3.3 Normalizasyon Çıktısı
Her belge sonunda `DocumentPage` nesnelerine indirgenir. Bu nesneler ingestion’ın geri kalanı için tek giriş formatıdır.

## 4. Vision Extraction

### 4.1 Neden Gerekli?
PDF belgelerde:

- tablolar
- madde listeleri
- şematik yapılar
- imza, not ve görsel kutular

düz metin extraction ile çoğu zaman kaybolur. Bu nedenle sayfa görselinden ikinci bir içerik katmanı çıkarılır.

### 4.2 Çalışma Şekli
Sayfa PNG olarak modele gönderilir. Modelden beklenen çıktı:

- `has_visual_content`
- `tables`
- `figures`
- `additional_text`

### 4.3 Dayanıklılık Problemi
Model JSON’u her zaman düzgün döndürmez. Bu yüzden LLM parse katmanında yarım JSON, trailing comma ve unterminated string gibi durumlar için toleranslı bir onarım katmanı eklenmiştir.

### 4.4 Tasarım Kararı
Vision extraction başarısız olsa bile ingest devam eder. Çünkü vision kritik ama vazgeçilmez değildir. Bu karar, kullanıcı deneyiminde tam çöküş yerine sınırlı değer kaybı üretir.

## 5. Semantic Chunking

### 5.1 Neden Basit Fixed-Size Chunking Yetmez?
Sabit karakter veya sabit kelime bazlı bölme şu sorunları üretir:

- cümle ortasında kesme
- bölüm bağlamını kaybetme
- gereksiz tekrar
- çok kısa veya çok uzun parçalar

Bu nedenle proje çok katmanlı bir semantic chunking tasarımı kullanır.

### 5.2 Katmanlar

#### Katman 1: Structural Parse
Başlıklar regex ile tespit edilir. Amaç, metni sadece ardışık cümleler olarak değil, belge hiyerarşisi içinde okumaktır.

#### Katman 2: Sentence Segmentation
Chunk sınırları cümle bütünlüğünü bozmayacak şekilde belirlenir.

#### Katman 3: Semantic Boundary Detection
Ardışık cümleler embed edilerek konu geçişlerinin olduğu noktalarda kırılma adayı oluşturulur.

#### Katman 4: Token-Aware Sizing
Chunk’lar embedding modelinin ve retrieval kalitesinin uygun gördüğü token aralığında tutulur.

#### Katman 5: Context Enrichment
Breadcrumb ve gerekirse kısa bağlamsal özet chunk başına eklenir.

## 6. Cross-Page Section Chunking
Yeni tasarımda chunking sadece tek sayfaya bağlı değildir. Aynı başlık veya breadcrumb ile devam eden ardışık sayfalar birleştirilebilir. Bu yaklaşım şu sorunu çözmeyi hedefler:

- başlık sayfa 10’da kalır
- asıl açıklama sayfa 11’de devam eder
- klasik page-local chunking iki zayıf chunk üretir

Bu yeni yapı sayesinde mantıksal bölüm korunur.



## 7. Contextual Enrichment
Contextual enrichment, chunk metninin üstüne belge bağlamını özetleyen kısa bir LLM açıklaması ekler. Amaç, embedding’in yalnızca lokal cümleleri değil, chunk’ın belge içindeki rolünü de taşımasıdır.

Örnek faydalar:

- "14 gün" ifadesinin yıllık izin mi, ihbar mı, eğitim süresi mi olduğunu daha iyi ayırt etmek
- aynı kavramın farklı bölümlerde geçtiği durumlarda section bilgisini güçlendirmek

Bu özellik kontrollü şekilde uygulanır:

- yalnızca yeterince büyük chunk’larda
- sınırlı uzunlukta
- yeni bilgi uydurmadan

## 8. DocumentIndexer

### 8.1 Sorumlulukları

- belgeyi işlemek
- text chunk’ları üretmek
- visual chunk’ları üretmek
- raw koleksiyona yazmak
- relatives üretmek
- relatives koleksiyona yazmak

### 8.2 Paralel Çalışma
Vision extraction ve relatives generation thread pool ile paralel yürütülür. Bu, ingestion süresini azaltır fakat aynı anda daha fazla LLM çağrısı üretir. Bu nedenle:

- maliyet
- rate limit
- hata yönetimi

önemli hale gelir.

### 8.3 Relatives Kalite Filtresi
Modelin ürettiği her soru kaydı doğrudan index’e yazılmaz. Yeni filtreler şunları eler:

- çok kısa sorular
- aşırı genel sorular
- chunk ile yeterince örtüşmeyen sorular
- tekrar eden varyantlar

Bu filtreleme, relatives koleksiyonunun gürültü üretmesini azaltır.

## 9. Index Tasarımı

### 9.1 Raw Index
Gerçek chunk içeriğini tutar.

### 9.2 Relatives Index
Chunk’lardan türetilmiş kullanıcı soru varyantlarını tutar.

### 9.3 Metadata Zenginliği
Metadata içinde yalnızca sayfa ve kaynak adı değil, ayrıca:

- breadcrumb
- token count
- parent text
- content type
- page_end

gibi retrieval ve generation için değerli alanlar tutulur.

## 10. Ingestion Katmanında Çözülmesi Gereken Problemler

- vision başarısızlıkları için ikinci deneme stratejisi
- OCR benzeri düşük kaliteli sayfalarda fallback parse
- büyük belgelerde maliyet kontrollü ingestion
- relatives üretimi için kalite skoru veya confidence ölçümü
- incremental re-indexing

## 11. Sonuç
Ingestion katmanı, bu projenin sadece veri hazırlama aşaması değil, retrieval başarısını belirleyen çekirdek mühendislik bileşenidir. Chunk kalitesi, metadata zenginliği ve relatives disiplini ne kadar güçlenirse sistemin verdiği cevapların doğruluğu o kadar artar.
