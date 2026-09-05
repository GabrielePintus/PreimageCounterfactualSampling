# Parallelizzazione delle proiezioni: piano di valutazione

## Obiettivo

Valutare la parallelizzazione intra-query delle proiezioni sulle regioni recuperate
dalla ricerca top-$k$ e aggiornare in modo trasparente i tempi online riportati nel
paper.

La modifica non interessa la costruzione dell'atlante. Per una query, le proiezioni
sulle regioni top-$k$ risolvono problemi indipendenti e possono quindi essere eseguite
contemporaneamente. I risultati sono ridotti nell'ordine deterministico degli anchor.
Poiché processi distinti possono produrre soluzioni numericamente diverse in presenza
di raffinamento della sparsità o decoding categorico, la regione vincente viene
risolta nuovamente nel processo principale prima di restituire il counterfactual.

L'equivalenza dei risultati non deve però essere soltanto assunta. Prima di riutilizzare
le vecchie metriche, i counterfactual prodotti dalle due implementazioni devono essere
confrontati query per query, soprattutto nei benchmark con raffinamento della sparsità
e decodifica categorica.

## Stato di avanzamento

Il benchmark sintetico completo della parallelizzazione è stato eseguito il
5 settembre 2026.

- Stato: **completo**.
- Query: 1.000.
- Configurazioni per query: 32.
- Misure complessive: 32.000.
- Valori di $k$: 1, 2, 3, 4, 5, 6, 7 e 8.
- Worker richiesti: 1, 2, 4 e 8.
- Backend: processi con pool persistente e warm-up escluso dai tempi.
- Parallelismo fra query: 1.
- Thread OpenMP, OpenBLAS, MKL e NumExpr: 1 per processo.
- Durata complessiva: 7 minuti e 44 secondi.
- Successo e target validity: 100%.
- Corrispondenza della distanza seriale/parallela: 24.000/24.000 esecuzioni
  parallele.
- Corrispondenza dell'anchor seriale/parallela: 24.000/24.000 esecuzioni
  parallele.
- Fallback: 0/32.000.

Hardware e ambiente:

- Intel Core i7-14700K, 20 core fisici e 28 CPU logiche;
- Linux 7.2.2;
- Python 3.11.15;
- CVXPY 1.8.2 e Clarabel 0.11.1.

Risultati principali confrontando l'esecuzione seriale con 8 worker:

| $k$ | Mediana seriale (ms) | Mediana, 8 worker (ms) | p95, 8 worker (ms) | Speedup mediano |
|---:|---:|---:|---:|---:|
| 1 | 3.712 | 3.695 | 6.205 | 1.001 |
| 2 | 6.708 | 4.035 | 7.414 | 1.654 |
| 3 | 9.703 | 4.099 | 6.696 | 2.376 |
| 4 | 12.680 | 4.171 | 7.093 | 3.055 |
| 5 | 15.657 | 4.284 | 6.879 | 3.675 |
| 6 | 18.618 | 4.613 | 7.002 | 4.137 |
| 7 | 21.618 | 5.512 | 7.015 | 4.054 |
| 8 | 24.597 | 6.124 | 7.162 | 4.166 |

Artifact definitivi:

- `results/topk_heuristic_ablation/topk_parallel_queries.parquet`;
- `results/topk_heuristic_ablation/topk_parallel_summary.parquet`;
- `results/topk_heuristic_ablation/topk_parallel.json`.

Il pilot originale da 50 query è conservato separatamente in
`results/topk_heuristic_ablation/pilot_50/`.

### Benchmark tabulare principale

È stato inoltre completato il confronto causale tra la versione seriale e quella
parallela corrente sulle 4.773 query dei sette dataset tabulari. Entrambi i rami
usano lo stesso codice, seed, campionamento casuale degli anchor, atlante da 500
regioni per classe, top-$5$, vincoli e configurazione di CLARABEL; differiscono
soltanto per `candidate_parallelism=1` e `candidate_parallelism=8`.

- successo e raggiungimento della classe target: 4.773/4.773 per entrambe;
- fallback oltre le regioni top-$5$: 0/4.773;
- stesso counterfactual entro $10^{-6}$: 4.772/4.773;
- stessa regione finale: 4.772/4.773;
- tempo online totale: 1.207,54 s seriale e 832,43 s parallelo;
- tempo medio: 253,0 ms seriale e 174,4 ms parallelo;
- mediana: 151,7 ms seriale e 79,1 ms parallelo;
- 95-esimo percentile: 801,5 ms seriale e 492,8 ms parallelo;
- speedup sul tempo complessivo: $1,45\times$;
- speedup mediano appaiato per query: $2,26\times$;
- query più veloci con la versione parallela: 93,44\%.

Tempi medi per dataset:

| Dataset | Seriale (ms) | Parallelo (ms) | Speedup del tempo totale |
|---|---:|---:|---:|
| Adult | 440,6 | 329,7 | $1,34\times$ |
| COMPAS | 165,3 | 87,2 | $1,90\times$ |
| German Credit | 1.663,1 | 1.621,4 | $1,03\times$ |
| Give Me Some Credit | 88,0 | 42,1 | $2,09\times$ |
| HELOC | 161,0 | 79,1 | $2,04\times$ |
| Lending Club | 238,5 | 159,9 | $1,49\times$ |
| Wisconsin Breast Cancer | 200,8 | 101,9 | $1,97\times$ |

L'unica differenza riguarda la query 1348 di Lending Club. Due regioni hanno
punteggi di selezione molto vicini e le soluzioni process-local di CLARABEL ne
invertono l'ordine. Entrambi i risultati sono target-validi; il ramo parallelo ha
$L_1$ inferiore di 0,0928, $L_2$ inferiore di 0,2475 e modifica una feature in più.
Di conseguenza, le medie globali cambiano rispettivamente di $-1,94\cdot10^{-5}$,
$-5,18\cdot10^{-5}$ e $8,38\cdot10^{-6}$ per $L_1$, $L_2$ e sparsità normalizzata.
Questo caso va riportato come variabilità numerica del solver, non come perfetta
identità degli output.

Artifact:

- `results/final_benchmark_certcf_serial_current.parquet`;
- `results/final_benchmark_certcf_parallel.parquet`;
- `results/final_benchmark_certcf_serial_parallel_comparison.parquet`;
- `results/final_benchmark_certcf_serial_parallel_comparison.json`.

## Evidenza preliminare

Il pilot sintetico comprende 50 query, valori $k=1,\ldots,8$ e 1, 2, 4 e 8 worker.
Con il backend a processi ha ottenuto fino a circa $4.21\times$ di speedup per $k=8$.
In tutte le 1.600 configurazioni sono rimasti invariati sia l'anchor selezionato sia
la distanza dalla query. Il pilot è indicativo, ma non è il risultato da riportare
nel paper.

## Benchmark da rieseguire

### Priorità 1: necessari

1. **Benchmark completo della parallelizzazione — completato**

   - 1.000 query sintetiche fissate;
   - $k=1,\ldots,8$;
   - 1, 2, 4 e 8 worker;
   - backend a processi con pool persistente e warm-up;
   - una sola query alla volta, senza parallelismo annidato;
   - un solo thread BLAS/OpenMP per processo;
   - salvataggi parziali e ripresa dopo un'interruzione.

   Il benchmark deve misurare mediana, media e 95-esimo percentile del tempo,
   speedup, efficienza parallela, tempo wall-clock delle proiezioni, lavoro totale,
   numero di proiezioni, fallback ed equivalenza con l'esecuzione seriale.

2. **Benchmark tabulare principale — completato**

   Rieseguire soltanto CertCF sulle 4.773 query originali usando la configurazione
   selezionata e top-$5$. Non occorre rieseguire gli altri metodi, addestrare
   nuovamente i classificatori o ricostruire le metriche qualitative se i nuovi
   counterfactual risultano equivalenti.

   Confrontare query per query successo, classe target, anchor selezionato, vettore
   del counterfactual entro tolleranza numerica, distanze $L_0$, $L_1$ e $L_2$ e
   fattibilità. Il confronto è stato completato; è stata osservata una sola
   differenza numerica su 4.773 query, descritta sopra.

3. **Ablazione HELOC Anchor-Ball/Anchor-PGD/CertCF — completata**

   Rieseguire soltanto le 500 query di CertCF con top-$5$. Lasciare invariati i
   risultati Anchor-Ball, Anchor-PGD e il build time dell'atlante. Ricalcolare il
   tempo online di CertCF e i rapporti di velocità con PGD, sia senza sia con
   ammortamento della costruzione dell'atlante.

   Il 5 settembre 2026 sono state rieseguite le 500 query HELOC con 8 processi
   richiesti, fino a 5 effettivamente utilizzabili per il top-$5$, pool persistente
   e warm-up di 53 ms escluso dai tempi. L'atlante esistente è stato caricato dal
   disco e non ricostruito. Non è mai stato necessario il fallback oltre top-$5$.

   Risultati su tutte le 500 query:

   | Misura CertCF | Seriale | Parallelo |
   |---|---:|---:|
   | Successo e target validity | 500/500 | 500/500 |
   | Tempo totale | 6,956 s | 2,212 s |
   | Tempo medio | 13,91 ms | 4,42 ms |
   | Mediana | 14,30 ms | 3,79 ms |
   | 95-esimo percentile | 14,69 ms | 6,23 ms |

   Lo speedup sul tempo totale è $3,15\times$, quello mediano appaiato è
   $3,48\times$ e 499 query su 500 sono più veloci in parallelo.

   Per coerenza con la tabella dell'ablazione, le metriche secondarie sono state
   ricalcolate sulle 368 query in cui tutti e tre i metodi hanno successo:

   | Metrica | Anchor-Ball | Anchor-PGD | CertCF parallelo |
   |---|---:|---:|---:|
   | Mean $L_1$ | 11,2017 | 11,2017 | 11,8417 |
   | Mean $L_2$ | 3,4141 | 3,4387 | 3,5902 |
   | Mean $L_0$ | 16,3424 | 16,3424 | 16,3043 |
   | Mean $\log_{10}(\mathrm{LOF})$ | 0,0243 | 0,0249 | 0,0224 |
   | Mean Isolation Forest score | -0,4137 | -0,4145 | -0,4117 |
   | Empirical target retention | 92,57\% | 92,90\% | 98,25\% |
   | All perturbations preserve target | 73,51\% | 74,86\% | 90,83\% |
   | Mean online time | 10,43 ms | 10,932 s | 4,37 ms |

   Le proiezioni parallele selezionano la stessa regione finale in tutte le 500
   query. Le soluzioni di CLARABEL differiscono numericamente fino a
   $5,44\cdot10^{-4}$, ma le metriche aggregate rimangono invariate alle cifre
   riportate nel paper; anche le due statistiche di robustezza empirica coincidono.

   Sulle query condivise Anchor-PGD è circa $2.500\times$ più lento online di
   CertCF. Includendo l'intero build time una tantum dell'atlante, 27,323 s,
   ammortizzato su sole 500 query, CertCF costa circa 59,1 ms per query e rimane
   circa $185\times$ più veloce di PGD.

   I precedenti JSON seriali e le tabelle aggregate sono conservati in
   `results/lirpa_refinement_ablation/archive/heloc_certcf_serial_before_parallel_2026-09-05/`.
   I risultati aggiornati sono in
   `results/lirpa_refinement_ablation/cases/heloc/certcf/queries/`,
   `results/lirpa_refinement_ablation/lirpa_refinement_queries.parquet`,
   `results/lirpa_refinement_ablation/empirical_robustness.parquet` e
   `results/lirpa_refinement_ablation/lirpa_refinement_summary.parquet`.

### Priorità 2: aggiornamento completo della sezione di scalabilità

4. **Griglia FCNN sintetica — completata**

   Il 5 settembre 2026 sono state rieseguite 1.000 query per ciascuna delle 25
   architetture, con top-$3$, backend a processi e 8 worker richiesti. Le 25.000
   query hanno avuto successo in entrambe le versioni e tutti i counterfactual
   paralleli coincidono esattamente con quelli seriali salvati.

   | Misura aggregata | Seriale | Parallelo |
   |---|---:|---:|
   | Tempo totale | 3.431,50 s | 2.553,72 s |
   | Tempo medio | 137,26 ms | 102,15 ms |
   | Mediana | 126,06 ms | 87,04 ms |
   | 95-esimo percentile | 207,72 ms | 172,05 ms |

   Lo speedup sul tempo totale è $1,34\times$, mentre la mediana degli speedup
   appaiati è $1,45\times$; l'85,18\% delle query è più veloce in parallelo. Lo
   speedup sul tempo totale varia poco fra le architetture, da $1,33\times$ a
   $1,36\times$, confermando che il tempo online rimane sostanzialmente
   indipendente dalla dimensione della rete anche nella nuova implementazione.

   Gli artifact seriali sono conservati in
   `results/network_complexity/archive/serial_before_parallel_2026-09-05/`. I
   risultati paralleli e il confronto appaiato sono in
   `results/network_complexity/network_complexity_queries.parquet`,
   `results/network_complexity/parallel_rerun_pairs.parquet`,
   `results/network_complexity/parallel_rerun_summary.parquet` e
   `results/network_complexity/parallel_rerun_comparison.json`.

5. **MNIST con LeNet-5 — completato**

   Il 5 settembre 2026 sono state rieseguite le 1.000 query targeted con top-$3$,
   8 processi richiesti e pool persistente. Il vecchio benchmark non serializzava
   l'atlante, quindi è stato necessario ricostruirlo; il nuovo build ha richiesto
   1.695,9 s, contro i 1.741,6 s originali. Questa differenza non viene attribuita
   alla parallelizzazione delle proiezioni.

   | Misura online | Seriale originale | Parallelo |
   |---|---:|---:|
   | Successo e target validity | 1.000/1.000 | 1.000/1.000 |
   | Tempo totale | 34,83 s | 27,14 s |
   | Tempo medio | 34,83 ms | 27,14 ms |
   | Mediana | 31,65 ms | 25,94 ms |
   | 95-esimo percentile | 56,32 ms | 30,93 ms |

   Lo speedup sul tempo totale è $1,28\times$, la mediana degli speedup appaiati
   è $1,22\times$ e il 97,4\% delle query è più veloce nella nuova esecuzione.
   Successo e classe target coincidono per tutte le query. I counterfactual sono
   identici entro $10^{-6}$ in 998 casi su 1.000. Nei due casi rimanenti il nuovo
   build numericamente indipendente ha prodotto una diversa geometria/ordinamento
   locale dell'atlante per il target 4; non è quindi corretto attribuire tali due
   differenze alla sola esecuzione parallela. Le medie cambiano lievemente:
   $L_1$ da 80,3807 a 80,3883, $L_2$ da 7,53651 a 7,53698 e sparsità normalizzata
   da 0,217446 a 0,217466.

   Artifact:

   - `results/mnist_certcf_lenet5_parallel.parquet`;
   - `results/mnist_certcf_lenet5_parallel_comparison.parquet`;
   - `results/mnist_certcf_lenet5_parallel_comparison.json`.

6. **CIFAR-10 con ResNet-20/32/56 — completato**

   È stato prima eseguito il pilot ResNet-20 sulle 10 query salvate, riutilizzando
   l'atlante: il tempo mediano è sceso da 1,092 s a 0,448 s, con speedup aggregato
   $2,30\times$. Poiché il vantaggio era netto, sono state quindi rieseguite tutte
   le 300 query delle tre reti, senza ricostruire gli atlanti.

   | Rete | Query | Seriale medio (s) | Parallelo medio (s) | Seriale mediano (s) | Parallelo mediano (s) | Speedup totale |
   |---|---:|---:|---:|---:|---:|---:|
   | ResNet-20 | 100 | 1,087 | 0,445 | 1,073 | 0,434 | $2,44\times$ |
   | ResNet-32 | 100 | 1,066 | 0,465 | 1,059 | 0,437 | $2,29\times$ |
   | ResNet-56 | 100 | 1,071 | 0,437 | 1,067 | 0,436 | $2,45\times$ |

   Complessivamente, il tempo online è sceso da 322,46 s a 134,73 s: media da
   1,075 s a 0,449 s, mediana da 1,065 s a 0,435 s e 95-esimo percentile da
   1,119 s a 0,551 s. Lo speedup aggregato è $2,39\times$, quello mediano
   appaiato $2,45\times$, e tutte le 300 query sono più veloci. Entrambe le
   versioni hanno 300/300 successi, selezionano lo stesso anchor e producono
   counterfactual uguali entro $1,32\cdot10^{-6}$. I tempi originali di
   costruzione degli atlanti restano validi.

   Artifact:

   - `results/cifar_resnet_scaling/cifar_resnet_queries.parquet`;
   - `results/cifar_resnet_scaling/cifar_resnet_summary.parquet`;
   - `results/cifar_resnet_scaling/parallel_rerun_pairs.parquet`;
   - `results/cifar_resnet_scaling/parallel_rerun_summary.parquet`;
   - `results/cifar_resnet_scaling/parallel_rerun_comparison.json`.

### Priorità 3: soltanto se mantenuti nel paper

- **Ablazione fra obiettivi $L_1$ e $L_2$ — completata.** Sono state
  rieseguite 9.546 query, 4.773 per obiettivo, con top-$5$ e 8 processi. Tutte
  raggiungono il target. Per $L_1$, il tempo totale scende da 153,22 s a
  138,64 s ($1,11\times$), la mediana globale da 15,75 ms a 10,63 ms e la
  media delle sette mediane da 24,64 ms a 13,67 ms. Per $L_2$, il tempo totale
  scende da 341,51 s a 275,45 s ($1,24\times$), la mediana globale da 17,49 ms
  a 11,56 ms e la media delle mediane da 27,03 ms a 23,79 ms. I build sono
  numericamente indipendenti, quindi il confronto non va descritto come
  equivalenza causale query per query; le conclusioni qualitative restano però
  stabili. La variazione macro-media per dataset del ramo $L_2$ rispetto a
  $L_1$ è +4,54\% in $L_1$, -7,53\% in $L_2$ e +8,29\% in $L_0$.

  Artifact principali:

  - `results/certcf_query_norm_ablation_parallel.parquet`;
  - `results/certcf_query_norm_l1_parallel_comparison.json`;
  - `results/certcf_query_norm_l2_parallel_comparison.json`.

- **Ramo CertCF del confronto ausiliario con VeriX — completato senza rebuild.**
  Le 250 coppie architettura--query sono un sottoinsieme esatto della griglia
  FCNN parallela, eseguita con lo stesso top-$3$. Successo e classe predetta
  coincidono in 250/250 casi e la massima differenza $L_1$ è
  $2,82\cdot10^{-6}$. Sui 55 successi condivisi con VeriX, il tempo medio
  CertCF passa da 139,31 ms a 100,54 ms, con speedup del tempo totale
  $1,39\times$. VeriX rimane invariato a 34,48 s medi sui successi condivisi.
  L'aggiornamento derivato conserva le metriche originali e sostituisce soltanto
  i tempi CertCF verificati.

  Artifact:

  - `results/verix_certcf_synthetic32_grid/verix_certcf_queries_parallel.parquet`;
  - `results/verix_certcf_synthetic32_grid/certcf_parallel_timing_comparison.json`.

- **Audit degli altri esperimenti con query time e $k>1$ — completato.** Gli
  esperimenti destinati all'appendice che richiedevano nuove misure erano il
  benchmark tabulare principale, l'ablazione HELOC, la griglia FCNN, MNIST,
  CIFAR, l'ablazione $L_1/L_2$ e il ramo CertCF del confronto VeriX; sono tutti
  coperti sopra. Il benchmark top-$k$ dispone già delle misure seriali e
  parallele dedicate. Non occorre raccogliere altri dati a causa di questa
  modifica.

Non è necessario rieseguire training, costruzione LiRPA degli atlanti o baseline che
non usano la nuova implementazione.

## Modifiche previste al paper

### Appendice: parallel query-time projection

Aggiungere alla sezione sulla scalabilità una sottosezione dedicata. Per le regioni
recuperate $P_i$, le proiezioni

$$
x_i^\star = \arg\min_{x \in P_i \cap \mathcal{C}(x_q)} d(x,x_q)
$$

sono indipendenti. La loro esecuzione concorrente non cambia il problema matematico
né la selezione finale, ma riduce la latenza wall-clock usando più core.

Se $P$ è il costo di una proiezione, il lavoro rimane approssimativamente $O(kP)$.
Con $W$ worker, il tempo ideale è

$$
O\left(\left\lceil\frac{k}{W}\right\rceil P\right)
$$

oltre all'overhead di coordinamento. Non si deve quindi sostenere che sia stata
eliminata la dipendenza da $k$.

Riportare:

- tempo rispetto a $k$ per 1, 2, 4 e 8 worker;
- speedup ed efficienza parallela;
- mediana e 95-esimo percentile;
- percentuale di equivalenza con l'esecuzione seriale;
- hardware, numero di core, backend e configurazione dei thread numerici.

### Tabelle e testo esistenti

Aggiornare i tempi del benchmark tabulare, dell'ablazione HELOC, della griglia FCNN,
di MNIST e, se il pilot lo giustifica, di CIFAR. Conservare nella nuova sottosezione
anche il confronto seriale/parallelo, invece di sostituire silenziosamente i vecchi
valori.

Nel pseudocodice online, esprimere il ciclo sulle regioni recuperate come un ciclo
parallelo, seguito dalla riduzione deterministica. Nella parte principale è sufficiente
una frase breve che segnali l'esecuzione concorrente delle proiezioni top-$k$.

## Protocollo di reporting

Tenere sempre distinti:

- tempo di costruzione dell'atlante, invariato e una tantum;
- latenza online seriale;
- latenza online parallela;
- numero di worker effettivi;
- throughput, qualora venga misurato.

La parallelizzazione intra-query mira alla latenza della singola query. Usa più CPU e
non implica automaticamente un throughput maggiore quando più query sono servite
contemporaneamente.

## Ordine operativo

1. Completare il benchmark parallelo da 1.000 query. **Completato.**
2. Verificare equivalenza e integrità dei risultati. **Completato.**
   La stabilità statistica e le visualizzazioni saranno finalizzate prima
   dell'inserimento nel paper.
3. Rieseguire il solo ramo CertCF del benchmark tabulare. **Completato.**
4. Aggiornare l'ablazione HELOC. **Completato.**
5. Rieseguire griglia FCNN, MNIST e infine il pilot CIFAR. **Completato, incluse
   tutte le 300 query CIFAR.**
6. Aggiornare valori, grafici e testo del paper solo dopo la validazione. **La
   raccolta e la validazione sono complete; questo è il prossimo passo.**
