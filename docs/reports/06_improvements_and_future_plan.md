# 06. Geliştirilmesi Gereken Yönler ve Gelecek Planı



## 1. Mevcut Durumun Kısa Özeti
Sistem şu anda çalışan bir RAG mimarisi sunmaktadır. Doküman ingest etme, chunk üretme, retrieval, grounding, cevap üretme ve kaynak gösterme zinciri çalışmaktadır. Ancak bu durum, sistemin üretim kalitesine ulaştığı anlamına gelmez.

Mevcut durumda sistem:

- temel retrieval ihtiyaçlarını karşılayabiliyor,
- source-aware cevap üretebiliyor,
- fallback stratejileri ile recall artırabiliyor,
- UI üzerinden kaynak gösterimi sunabiliyor.

Buna karşılık aşağıdaki alanlar hâlâ gelişime açıktır:

- retrieval hassasiyeti
- citation doğruluğu
- ingest dayanıklılığı
- observability
- test derinliği
- belge yaşam döngüsü yönetimi

## 2. Geliştirilmesi Gereken Ana Alanlar

## 2.1 Retrieval Kalitesinin Artırılması
Bugünkü sistemde retrieval davranışı embedding tabanlı similarity ve sınırlı lexical re-ranking ile yönetilmektedir. Bu iyi bir başlangıçtır ancak özellikle karmaşık sorularda daha güçlü sıralama mekanizmalarına ihtiyaç vardır.

### Sorunlar

- bazı sorgularda semantik olarak yakın ama içerik olarak zayıf chunk’lar üst sıralara çıkabiliyor
- benzer bölümler arasında ayrım yapmak zorlaşabiliyor
- tek threshold tüm soru tipleri için ideal çalışmıyor

### Hedef

- daha doğru top-k sonuç
- daha düşük false positive grounded cevap oranı
- daha iyi source relevance



## 2.2 Citation Doğruluğunun Güçlendirilmesi
Şu anda cevap içinde citation üretimi ile UI’da sunulan source listesi arasında iyi bir bağ vardır; ancak bu ilişki henüz sentence-level deterministik değildir.

### Sorunlar

- cevap içindeki citation ile gerçek destekleyici chunk arasında birebir bağ her zaman garanti değil
- aynı sayfada birden fazla chunk olduğunda destekleyici cümle seviyesi netleşmeyebiliyor
- kullanıcı "bu bilgi tam olarak hangi parçadan geldi" sorusuna her zaman en net cevabı alamıyor

### Hedef

- sentence-to-source alignment
- deterministic citation mapping
- kullanıcı güveninin artırılması



## 2.3 Ingestion Dayanıklılığı ve Vision Pipeline Stabilitesi
Vision extraction ve relatives generation, projeye değer katsa da hata üretmeye en açık alanlardır.

### Sorunlar

- LLM çıktılarında bozuk JSON oluşabiliyor
- bazı sayfalarda vision extraction tutarlı sonuç vermiyor
- ingest sürecinde hata olmasa da kalite kaybı oluşabiliyor
- büyük dokümanlarda maliyet ve gecikme büyüyor

### Hedef

- daha stabil ingestion
- görsel ağırlıklı sayfalarda daha güvenilir veri çıkarımı
- büyük dosyalarda daha öngörülebilir performans



## 2.4 Chunk Kalitesinin Daha da Güçlendirilmesi
Semantic chunking ve cross-page section merge eklenmiş olsa da chunk kalitesi sürekli optimize edilmesi gereken bir alandır.

### Sorunlar

- bazı chunk’lar hâlâ fazla geniş veya fazla dar olabilir
- section merge mantığı her belge tipi için ideal çalışmayabilir
- contextual enrichment bazı belgelerde faydalı, bazılarında gürültü üretebilir

### Hedef

- daha tutarlı chunk boyutu
- daha yüksek retrieval precision
- daha temiz semantic boundaries


## 2.5 Observability ve Sistem İzlenebilirliği
Şu anda sistem çalışıyor olsa da üretim seviyesi tanılama için log ve metrik yüzeyi sınırlıdır.

### Sorunlar

- neden belirli fallback stratejisinin seçildiği net görünmeyebilir
- grounded=false kararının nedeni sistematik olarak izlenmiyor
- ingestion sürecindeki kalite kayıpları tek bakışta görülemiyor

### Hedef

- query trace görünürlüğü
- retrieval ve generation kararlarının açıklanabilir logları
- sistem sağlığını izlemeye uygun dashboard verisi



## 2.6 Test Kapsamının Genişletilmesi
Mevcut testler önemli yapı taşlarını koruyor, ancak daha gerçekçi senaryolar için daha geniş bir test çerçevesi gerekiyor.

### Sorunlar

- uçtan uca gerçek belge-soru-citation senaryoları sınırlı
- regresyon riskleri prompt değişimlerinde sessiz kalabiliyor
- retrieval benchmark seti henüz yeterince kapsamlı değil

### Hedef

- davranışsal regresyonların erken yakalanması
- retrieval tuning çalışmalarının sayısal ölçümü
- daha güvenli sürüm geçişleri



## 2.7 Doküman Versiyonlama ve Yaşam Döngüsü Yönetimi
Kurumsal dokümanlar zaman içinde değişir. Bu nedenle sadece index oluşturmak yetmez; dokümanın yaşam döngüsünü de yönetmek gerekir.

### Sorunlar

- güncellenmiş doküman ile eski chunk’ların karışması mümkün
- kullanıcı eski politika bilgisini alabilir
- silme ve yeniden ingest süreçlerinde tutarlılık riski vardır

### Hedef

- version-aware index yapısı
- aktif / pasif belge ayrımı
- zaman bazlı geçerlilik kontrolü



## 2.8 UI ve Açıklanabilirlik Deneyimi
Sistemin verdiği doğru cevap kadar, bunu kullanıcıya nasıl anlattığı da önemlidir.

### Sorunlar

- kullanıcı bazı durumlarda ana kaynak ile ek bağlam ayrımını ilk bakışta anlamayabilir
- neden belirli sayfaların geldiği daha açık gösterilebilir
- evidence preview daha güçlü hale getirilebilir

### Hedef

- daha anlaşılır source sunumu
- daha yüksek kullanıcı güveni
- daha iyi debug deneyimi



## 4. Önceliklendirme
Tüm geliştirmeler aynı önemde değildir. Projenin kısa vadede en kritik alanları aşağıdaki sırayla ele alınmalıdır:

1. Ingestion dayanıklılığı ve vision/relatives stabilitesi
2. Retrieval kalitesinin artırılması
3. Citation doğruluğunun iyileştirilmesi
4. Observability altyapısının kurulması
5. Test kapsamının genişletilmesi
6. Doküman versiyonlama
7. UI explainability iyileştirmeleri

Bu sıralamanın gerekçesi şudur:

- ingestion bozuksa retrieval zaten zayıflar
- retrieval zayıfsa generation güvenilir olmaz
- citation zayıfsa kullanıcı güveni azalır
- observability yoksa iyileştirme süreci yavaşlar

## 5. Planlanan değişimler

- ingest hata kayıtlarının analizi
- observability tasarımının çıkarılması
- retrieval başarısız örneklerinin toplanması
- cross-encoder reranking prototipi
- source/citation eşleştirme iyileştirmeleri
- golden set hazırlığı
- UI kaynak açıklama tasarımı
- adaptive threshold deneyleri
- citation verifier tasarımı
- end-to-end testlerin eklenmesi
- version model tasarımı
- metrics dashboard ilk sürümü
- sentence-level evidence mapping
- belge tipi bazlı chunking profilleri
- ingestion kalite raporu
- UI explainability bileşenleri
- version metadata uygulaması
- aktif/pasif doküman sürüm yönetimi
- retrieval benchmark setinin genişletilmesi
- sistem kararlılık ve performans iyileştirmeleri

## 6. Beklenen Sonuçlar
Bu plan uygulandığında sistemde aşağıdaki kazanımlar beklenmektedir:

- daha doğru retrieval
- daha güvenilir citation yapısı
- daha stabil ingest süreci
- daha iyi gözlemlenebilirlik
- daha sağlam test kapsamı
- doküman güncelliği açısından daha güvenli kullanım
- daha açıklanabilir kullanıcı deneyimi
