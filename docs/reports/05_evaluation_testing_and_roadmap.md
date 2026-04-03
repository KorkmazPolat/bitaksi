# 05. Evaluation, Testing ve Yol Haritası 

## 1. Amaç
Bu rapor, sistemin nasıl ölçüldüğünü, nasıl test edildiğini ve sonraki iterasyonlarda hangi başlıklara odaklanılması gerektiğini açıklar.

## 2. Evaluation Neden Ayrı Bir Katman?
RAG sistemlerinde iyi görünen cevap, doğru retrieval yapıldığı anlamına gelmez. Aynı şekilde doğru retrieval yapılması da her zaman iyi bir kullanıcı cevabı üretileceğini garanti etmez. Bu nedenle evaluation iki katmanda ele alınır:

- retrieval kalitesi
- generation kalitesi

## 3. Retrieval Metrikleri

### 3.1 Recall@K
İlgili chunk’ların üst K sonuç içinde bulunup bulunmadığını ölçer.

### 3.2 MRR
İlk doğru sonucun sıralamadaki yerini ödüllendirir.

### 3.3 NDCG@K
Birden fazla ilgili sonucun sıralamadaki dağılımını değerlendirir.

## 4. Generation Metrikleri

### 4.1 Faithfulness
Cevaptaki iddiaların bağlam tarafından desteklenip desteklenmediğini ölçer.

### 4.2 Answer Relevancy
Cevabın kullanıcı sorusunu ne kadar iyi karşıladığını ölçer.

## 5. Test Stratejisi
Projede testler yalnızca yardımcı fonksiyonları değil, kritik davranışları da kapsar:

- semantic chunker testleri
- query strategy testleri
- smart grounding testleri
- response generator testleri
- retrieval metrics testleri
- llm parse utility testleri

### 5.1 Test Felsefesi
LLM ve vector store gibi dış bağımlılıkları doğrudan test etmek yerine davranış kontratlarını test etmek daha sürdürülebilir bir yaklaşımdır.

### 5.2 Sonradan Eklenen Kritik Testler

- cross-page section merge davranışı
- lexical reranking davranışı
- bozuk JSON onarımı
- tek güçlü chunk ile grounded cevap üretimi

## 6. İzlenebilirlik İhtiyacı
Mevcut sistem çalışsa da üretim seviyesi gözlemlenebilirlik henüz sınırlıdır. Aşağıdaki kayıtlar çok değerlidir:

- query id
- strategy used
- queries tried
- top score distribution
- grounded false reason
- ingest latency breakdown
- vision failure rate
- relatives generation success/failure rate

## 7. Yol Haritası

### 7.1 Kısa Vadeli

- citation verifier
- vision retry strategy
- ingest kalite raporu
- threshold tuning

### 7.2 Orta Vadeli

- cross-encoder reranker
- document versioning
- benchmark dataset
- retrieval traces dashboard

### 7.3 Uzun Vadeli

- human feedback loop
- active learning for relatives
- document lifecycle governance
- role-aware retrieval and access control


## 8. Sonuç
Bu proje sadece çalışan bir prototip değil, ölçülebilir ve iteratif biçimde geliştirilebilir bir RAG sistemidir. Değerlendirme, test ve yol haritası katmanları projeyi akademik ve mühendislik açısından daha güçlü hale getirir.
