# Teknik Rapor Seti

Bu klasör, projeyi tek bir raporda toplamak yerine alt sistemlere ayıran detaylı teknik raporları içerir.

## Rapor Listesi

1. `01_system_overview.md`
   Projenin genel mimarisi, veri akışı, modüller arası ilişkiler ve sistemin bütünsel tasarımı.

2. `02_ingestion_and_indexing.md`
   Doküman işleme, vision extraction, semantic chunking, contextual enrichment ve çoklu index yazımı.

3. `03_retrieval_and_grounding.md`
   Raw + relatives retrieval, fallback zinciri, lexical re-ranking ve grounded karar mekanizması.

4. `04_generation_api_and_ui.md`
   Cevap üretimi, API katmanı, chat kontratı, kaynak gösterimi ve arayüz davranışları.

5. `05_evaluation_testing_and_roadmap.md`
   Retrieval/generation metrikleri, test stratejisi, izlenebilirlik ihtiyaçları ve geliştirme yol haritası.

## Kullanım

- Genel bir proje anlatımı için önce `01_system_overview.md` okunmalıdır.
- Ingestion veya chunk kalitesi odaklı teknik analiz için `02_ingestion_and_indexing.md` tercih edilmelidir.
- Arama kalitesi ve grounding kararları için `03_retrieval_and_grounding.md` en önemli rapordur.
- Uygulama davranışı ve kullanıcı cevabı üretimi için `04_generation_api_and_ui.md` okunmalıdır.
- Sunum, değerlendirme ve gelecek iterasyon planlaması için `05_evaluation_testing_and_roadmap.md` kullanılmalıdır.
