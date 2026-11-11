# LunarRocket terrain makalesi — LaTeX paketi

Hedef dergi: **Autonomous Intelligent Systems** (Springer, ISSN 2730-616X).
Tam açık erişim, **APC yok** (Tongji Üniversitesi sponsorluğunda), CC BY 4.0,
Scopus + DOAJ, CiteScore 8,2. Derginin kendi kılavuzu Springer Nature LaTeX
şablonunu öneriyor; .docx de kabul ediliyor.

## İki derleme, tek içerik

| Dosya | Sınıf | Nerede derlenir | Ne için |
|---|---|---|---|
| `main.tex` | `article` (nötr) | yerel — `tectonic -X compile main.tex` | Prova okuma, sayfa sayımı, atıf kontrolü |
| `main_sn.tex` | `sn-jnl` (Springer) | yerel — `tectonic -X compile main_sn.tex` | **Dergiye gidecek gerçek dosya** |

`body.tex`, `backmatter.tex`, `abstract.tex`, `sections/`, `tables/`,
`figures/` ve `refs.bib` **ikisinde de aynen paylaşılır** — sınıf değiştirmek
için bu dosyalarda hiçbir değişiklik gerekmez. (Bu, HydroGym makalesindeki
`paper_tex` + `paper_jai` deseninin aynısı.)

### sn-jnl.cls artık pakette

`sn-jnl.cls` CTAN'da yok — yalnızca Springer Nature'ın kendi LaTeX paketiyle
dağıtılıyor. Sınıf dosyası ve `sn-mathphys-num.bst`, `rstudio/rticles`
deposundaki resmî Springer şablonundan alınıp bu klasöre kondu (LPPL lisanslı,
`\ProvidesClass{sn-jnl}`). Artık **Overleaf'e gerek yok**, paket yerelde
derleniyor:

```bash
tectonic -X compile main_sn.tex     # -> main_sn.pdf, 29 sayfa
```

Overleaf'i yine tercih ederseniz klasörü zip'leyip yükleyin, ana dosya
`main_sn.tex`.

### Preamble'dan çıkarılanlar

Springer iskeletinin `algorithm`/`algorithmicx`/`manyfoot`/`multirow` paketleri
ve `\theoremstyle{thmstyleone}` + `\newtheorem` satırları kaldırıldı: bu
makalede ne algoritma ne teorem ortamı var, ve `thmstyleone`
`sn-mathphys-num` seçeneği altında tanımsız — derlemeyi kırıyordu.

`\orcid` da bu sınıf sürümünde (v0.1, 2019) `.cls` içinde yorum satırı, yani
kullanılamıyor. ORCID artık tex'te bir not olarak duruyor; Springer zaten
ORCID'leri **Editorial Manager'da** topluyor ve yayımlanan kopya oradan
geliyor.

## Mevcut durum

Her iki derleme de temiz:

| Derleme | Sayfa | Çözülmemiş atıf | Eksik şekil | Kaynak |
|---|---|---|---|---|
| `main.tex` (nötr) | 25 | 0 | 0 | 30 |
| `main_sn.tex` (**Springer**) | 29 | 0 | 0 | 30 |

Springer dizgisi tek kolon ve iri puntolu olduğu için nötr derlemeden daha
uzun; AIS sert bir sayfa sınırı koymuyor.

`sn-mathphys-num` stili kitap kaynaklarında yayıncı şehrini zorunlu tutuyordu —
`ebert2002` girdisine `address` eklendi, aksi hâlde kaynakçada
"Kaufmann, ??? (2002)" basılıyordu. PDF'te artık hiç `??` yok.

Kaynakça **18'den 30'a** çıkarıldı (2026-10-07). Eklenen 12 kaynağın hepsi
OpenAlex üzerinden DOI'siyle doğrulandı ve metne anlamlı yerlere yerleştirildi
— yetim (atıfsız) kaynak yok, hayalet atıf yok; denetim:

```bash
grep -oP '^@\w+\{\K[^,]+' refs.bib | sort > /tmp/k.txt
grep -ohE '\\cite[a-z]*\{[^}]*\}' sections/*.tex | sed 's/.*{//;s/}//' \
  | tr ',' '\n' | sort -u > /tmp/c.txt
comm -3 /tmp/k.txt /tmp/c.txt     # ciktinin bos olmasi gerekir
```

Eklenen kaynakların dağılımı: LOD/çok-çözünürlüklü arazi (Pajarola & Gobbetti
2007 derlemesi, Zhang vd. 2019 dikiş şeritleri, Jain vd. 2024), partition of
unity'nin kanonik kaynağı (Ohtake vd. 2003, MPU implicits), RL tohum
varyansı/istatistiksel güç (Agarwal vd. 2021, Colas vd. 2018 ve 2019),
randomize simülasyon ve simülatör seçimi (Muratore vd. 2022, Collins vd.
2021), temas/terramekanik (Nakashima vd. 2010), ve en yakın komşu çalışmalar
(Richard vd. 2024 — OmniLRS, Scorsoglio vd. 2021 — görüntü tabanlı ay inişi
meta-RL).

**Yazar listesi** (2026-10-18, ORCID ve afiliasyonlarla): Cem Özkurt (sorumlu
yazar; Veri Bilimi ve Analitiği, Sakarya Üniv.), Muhammed Emin Onay ve Eyüp
Altuğ Tunç (Bilgisayar Müh., Sakarya Üniv.), Ahmet Kutay Küçükler ve Haktan
Yusuf Güloğlu (Bilgisayar Müh., Sakarya Uygulamalı Bilimler Üniv.). Sıra
HydroGym makalesiyle aynı. `../build_full_paper_en.py` .docx hattı da aynı
listeyi taşıyor.

## Dergi kurallarına uyum (2026-10-07)

Paket Springer Nature'ın zorunlu gereksinimlerine göre yeniden kuruldu:

| Kural | Gereksinim | Durum |
|---|---|---|
| Özet uzunluğu | 150-250 kelime | **195** (2026-10-18 ölçümü) |
| Anahtar kelime | 4-6 | **6** |
| Declarations | Funding, Competing interests, Ethics approval, Consent, Data availability, Code availability, Author contributions — kaynakçadan **önce** | **8 başlığın hepsi var** (`backmatter.tex`) |
| Şablon | Springer Nature LaTeX önerilir | `main_sn.tex` (`sn-jnl`) |
| Kaynak biçimi | Numaralı Springer stili | `sn-mathphys-num` |
| Tablolar | Arabic numaralı, ardışık, başlıklı | 6 tablo |
| Sayfa sınırı | Dergi sert sınır koymuyor | 25 (nötr, tek kolon) |

Beyanlar ayrı kaynak dosyalarında tutulur, böylece hem LaTeX hem .docx hattı
aynı metni kullanır: `14_funding`, `11_conflict_of_interest`, `15_ethics`,
`16_consent`, `12_data_availability`, `13_code_availability`,
`10_author_contributions`, `09_acknowledgements`.

`\bmhead` sn-jnl'in arka madde başlığıdır; `main.tex` nötr sınıf için bir
yedek tanım taşır, bu yüzden `backmatter.tex` iki derlemede de aynen çalışır.

## Gönderim öncesi kapatılacaklar

- [x] Sorumlu yazar e-postası: cemozkurt@sakarya.edu.tr
- [ ] Küçükler'in CRediT rolleri taslak (Investigation, Validation, Review & Editing) — kendisiyle teyit edilmeli
- [ ] ORCID'ler tex'te yorum olarak duruyor; Editorial Manager'da girilecek
- [ ] `sections/12_data_availability.tex`: "available from the author on
      reasonable request" diyor. Springer açık veriyi tercih eder; depoyu
      yayına açıp DOI vermek daha güçlü olur
- [ ] Başlık ve özet, null RL sonucuna göre gözden geçirilsin — şu anki başlık
      üç yöntemin karşılaştırmasını vaat ediyor, asıl sonuç ise "karıştırma
      şekli sınır sürekliliğini belirliyor, RL başarısını belirlemiyor"

## Dosyaları yeniden üretmek

`sections/` ve `tables/` içindeki dosyalar `../sections_en/*.txt` ve
`../tables_en/*.csv` kaynaklarından üretilir:

```bash
python3 build_tex.py
```

Betik LaTeX kaçışlarını, `\texttt` kimliklerini ve yazar-yıl atıflarının
`\citep`/`\citet`/`\citealp` dönüşümünü yapar, sonunda dönüşmeyen atıf
adaylarını listeler. **İki dosyayı bilerek atlar** ve üzerine yazmaz:

- `sections/04_2_terrain_blending.tex` — elle yazıldı, denklemleri taşıyor
- `abstract.tex` — `../abstract_en.txt`'den kopyalandı (250 kelime, Springer
  sınırı 150–250; **tam sınırda**, genişletilirse taşar)

İçerik değişikliği `../sections_en/` kaynaklarında yapılır, sonra betik
yeniden koşulur.
