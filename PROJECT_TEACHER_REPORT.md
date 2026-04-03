# HR Assistant Chatbot - Modüler Teknik Rapor İndeksi

Bu dosya artık tek başına tüm detayları taşımak yerine, proje için hazırlanan modüler teknik rapor setinin giriş noktasıdır. Böylece sistemin önemli bölümleri ayrı ayrı, daha derinlikli ve daha okunabilir biçimde dokümante edilmektedir.

## Rapor Yapısı

1. [Sistem Genel Bakış](/Users/polatkorkmaz/Documents/GitHub/bitaksi/docs/reports/01_system_overview.md)
   Projenin amacı, mimarisi, veri akışı, temel tasarım ilkeleri ve genel risk çerçevesi.

2. [Ingestion ve Indexing](/Users/polatkorkmaz/Documents/GitHub/bitaksi/docs/reports/02_ingestion_and_indexing.md)
   Belge işleme, vision extraction, semantic chunking, contextual enrichment, relatives generation ve Chroma index yapısı.

3. [Retrieval ve Grounding](/Users/polatkorkmaz/Documents/GitHub/bitaksi/docs/reports/03_retrieval_and_grounding.md)
   Raw ve relatives retrieval, fallback zinciri, lexical reranking ve grounded karar mantığı.

4. [Generation, API ve UI](/Users/polatkorkmaz/Documents/GitHub/bitaksi/docs/reports/04_generation_api_and_ui.md)
   Cevap üretim süreci, chat kontratı, kaynak sunumu, UI davranışı ve explainability yüzeyi.

5. [Evaluation, Testing ve Yol Haritası](/Users/polatkorkmaz/Documents/GitHub/bitaksi/docs/reports/05_evaluation_testing_and_roadmap.md)
   Metrikler, test stratejisi, observability ihtiyacı ve sonraki iterasyon planı.

6. [Rapor Seti README](/Users/polatkorkmaz/Documents/GitHub/bitaksi/docs/reports/README.md)
   Alt raporların ne amaçla yazıldığı ve hangi sırada okunması gerektiği.

## Neden Modüler Rapor?

Tek bir rapor içinde:

- ingestion detayları
- retrieval karar mantığı
- generation policy katmanı
- UI davranışı
- test ve yol haritası

aynı derinlikte anlatıldığında belge hızla büyür ve okunabilirlik düşer. Modüler yapı şu avantajları sağlar:

- her teknik katman kendi terminolojisiyle daha detaylı anlatılabilir
- UML ve akış diyagramları bölüm bazlı eklenebilir
- akademik, teknik ve sunum amaçlı kullanım ayrıştırılabilir
- gelecekte yalnızca ilgili rapor güncellenebilir

## Okuma Önerisi

Projeyi ilk kez okuyacak biri için önerilen sıra:

1. [01_system_overview.md](/Users/polatkorkmaz/Documents/GitHub/bitaksi/docs/reports/01_system_overview.md)
2. [02_ingestion_and_indexing.md](/Users/polatkorkmaz/Documents/GitHub/bitaksi/docs/reports/02_ingestion_and_indexing.md)
3. [03_retrieval_and_grounding.md](/Users/polatkorkmaz/Documents/GitHub/bitaksi/docs/reports/03_retrieval_and_grounding.md)
4. [04_generation_api_and_ui.md](/Users/polatkorkmaz/Documents/GitHub/bitaksi/docs/reports/04_generation_api_and_ui.md)
5. [05_evaluation_testing_and_roadmap.md](/Users/polatkorkmaz/Documents/GitHub/bitaksi/docs/reports/05_evaluation_testing_and_roadmap.md)

## Not

Bu rapor seti, mevcut kod tabanındaki son mimari değişikliklerle uyumlu olacak şekilde hazırlanmıştır. Özellikle:

- cross-page section chunking
- contextual enrichment
- relatives kalite filtresi
- lexical reranking
- kaynak gösterimi ve UI iyileştirmeleri

ayrı raporlar içinde detaylandırılmıştır.
