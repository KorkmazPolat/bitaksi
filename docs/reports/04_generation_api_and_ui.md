# 04. Generation, API ve UI 

## 1. Amaç
Bu rapor, retrieval sonucu bulunan bağlamın kullanıcıya cevap olarak nasıl dönüştürüldüğünü ve bu cevabın API ile UI üzerinden nasıl sunulduğunu anlatır.

## 2. Generation Katmanı

### 2.1 ResponseGenerator Sorumluluğu
`ResponseGenerator` şu görevleri üstlenir:

- retrieval sonucunu kontrol etmek
- bağlamı LLM’e uygun formata sokmak
- LLM ile cevap üretmek
- gerekirse cevabı yeniden yazdırmak
- kaynak listesi çıkarmak

### 2.2 Prompt Politikası
Generation promptu yalnızca bağlam içi cevap üretmeyi zorlar. Başlıca kurallar:

- bağlam dışına çıkma
- citation kullan
- gereksiz citation tekrar etme
- kısa ve doğrudan cevap ver
- referansa yönlendirip geçme

Bu prompt tasarımı, modeli "yardımcı ama serbest" değil, "kontrollü ve kanıt odaklı" davranmaya iter.

### 2.3 Context Selection
Tüm retrieval sonucu modele verilmez. En yüksek kaliteli ve çeşitli birkaç chunk seçilir. Bu kararın amacı:

- gereksiz citation çoğalmasını önlemek
- token maliyetini azaltmak
- modelin dikkatini dağıtmamak

### 2.4 Rewrite Adımı
İlk cevap düşük kaliteli veya sadece kaynak gösteren bir yönlendirme ise ikinci bir rewrite adımı çalıştırılır. Bu, kullanıcı deneyimini iyileştirir.

## 3. API Katmanı

### 3.1 Ana Endpoint’ler

- `POST /documents/ingest`
- `GET /documents/`
- `GET /documents/{doc_id}/chunks`
- `GET /documents/{doc_id}/file`
- `GET /documents/{doc_id}/page/{page_num}`
- `DELETE /documents/{doc_id}`
- `POST /chat/`
- `POST /evaluation/*`

### 3.2 Chat Kontratı
İstek:

- `query`
- `session_id`
- `history`

Cevap:

- `answer`
- `sources`
- `strategy_used`
- `grounded`
- `queries_tried`

Bu kontrat yalnızca kullanıcıya cevap döndürmek için değil, açıklanabilir bir debug yüzeyi sağlamak için de önemlidir.

## 4. UI Katmanı

### 4.1 Ana Davranışlar

- chat paneli
- source chip’leri
- inline citation tıklama
- sağ panelde belge/sayfa gösterimi
- fallback görünüm

### 4.2 Kaynak Sunumu
UI artık kaynakları ikiye ayırabilir:

- ana kaynaklar
- ek bağlam

Bu ayrım, cevabın gerçekten dayandığı sayfalarla retrieval sırasında bakılan yardımcı sayfaları ayırmak için önemlidir.

### 4.3 Drawer Davranışı
Belge paneli açıldığında sohbet tamamen kapanmamalıdır. Bu nedenle masaüstünde yan kolon düzeni, mobilde ise overlay yaklaşımı tercih edilir.


## 6. Güçlü Yönler

- response yapısı denetlenebilir
- citation gösterimi kullanıcı güvenini artırır
- grounded=false durumda güvenli davranış vardır
- UI, kaynakla cevap arasında görsel köprü kurar

## 7. Çözülmesi Gereken Problemler

- sentence-level citation alignment eksikliği
- UI’da neden bu kaynakların geldiğinin daha iyi açıklanması
- çok kaynaklı cevaplarda kaynakların önem sırası gösterimi
- source preview ile evidence highlighting doğruluğu

## 8. Sonuç
Generation, API ve UI birlikte düşünüldüğünde sistem sadece teknik olarak cevap üretmez; aynı zamanda bu cevabın kaynağını kullanıcıya anlatmaya çalışır. Bu, projeyi sıradan bir chat arayüzünden daha güçlü hale getirir.
