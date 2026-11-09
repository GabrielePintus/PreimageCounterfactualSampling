# Metodi counterfactual per estendere il benchmark di CertCF

Data della ricognizione: 7 novembre 2026.

Questa ricerca raccoglie 15 metodi pertinenti all'estensione sperimentale di CertCF, includendo baseline consolidate e lavori del 2024–2026. La selezione privilegia il benchmark tabulare e distingue i metodi post-hoc da quelli che addestrano congiuntamente classificatore e generatore. Per ciascun metodo sono riportati titolo, anno, ambito e una sintesi di tre frasi.

Due distinzioni orientano il confronto:

- **Model-agnostic non significa indipendente dalla modalità del dato**: un metodo può accettare diversi classificatori, ma essere progettato per dati tabulari.
- **COCO è un riferimento ambiguo**: probabilmente indica CROCO; è incluso anche COCOA, che è un metodo distinto.

## Metodi inizialmente individuati

### PROPLACE

**Titolo:** *Provably Robust and Plausible Counterfactual Explanations for Neural Networks via Robust Optimisation*.

**Anno:** 2024, atti di ACML 2023.

**Ambito:** principalmente dati tabulari e reti neurali.

Genera counterfactual plausibili che mantengono la validità sotto variazioni limitate dei parametri della rete neurale. Alterna la ricerca di un counterfactual con la ricerca di una perturbazione avversaria del modello, restringendo la ricerca a una regione plausibile costruita mediante punti di riferimento robusti. È un confronto interessante per CertCF sul rapporto tra plausibilità, garanzie e costo computazionale, ma certifica cambiamenti del **modello**, non perturbazioni dell'**input**. [Paper originale](https://proceedings.mlr.press/v222/jiang24a.html)

### MACE

**Titolo:** *Model-Agnostic Counterfactual Explanations for Consequential Decisions*.

**Anno:** 2020, AISTATS.

**Ambito:** formulazione generale per modelli codificabili logicamente; applicazioni tabulari con feature eterogenee.

Codifica modello, distanza e vincoli in formule logiche e usa solver SAT/SMT per cercare counterfactual. Una ricerca sulla soglia di distanza permette di ottenere validità e ottimalità della distanza entro una tolleranza specificata. Pur non essendo recente, è una baseline importante per confrontare CertCF con una ricerca formalmente vincolata eseguita nuovamente per ogni query. [Paper originale](https://proceedings.mlr.press/v108/karimi20a.html)

### PROBE

**Titolo:** *Probabilistically Robust Recourse: Navigating the Trade-offs between Costs and Robustness in Algorithmic Recourse*.

**Anno:** 2023, ICLR.

**Ambito:** recourse su dati tabulari.

Considera il rischio che un counterfactual perda validità quando le modifiche suggerite vengono eseguite in modo impreciso. Permette di specificare una probabilità desiderata di invalidazione e cerca un compromesso tra tale obiettivo, costo delle modifiche e raggiungimento della classe desiderata. La nozione di robustezza è vicina a quella di CertCF perché riguarda perturbazioni dell'input, ma è probabilistica rispetto a un modello di rumore. [Paper originale](https://arxiv.org/abs/2203.06768)

### CROCO

**Titolo:** *Generating Robust Counterfactual Explanations*.

**Anno:** 2023, ECML PKDD.

**Ambito:** dati tabulari.

Genera counterfactual robusti a variazioni delle feature del punto restituito. Gestisce esplicitamente il compromesso tra vicinanza alla query e robustezza, consentendo di richiedere un livello minimo di quest'ultima. È quindi particolarmente pertinente per verificare se CertCF offre vantaggi rispetto a metodi che ottimizzano direttamente la robustezza alle perturbazioni dell'input. [Paper originale](https://arxiv.org/abs/2304.12943)

### C-CHVAE

**Titolo:** *Learning Model-Agnostic Counterfactual Explanations for Tabular Data*.

**Anno:** 2020, WWW.

**Ambito:** specificamente tabulare, con variabili numeriche e categoriche.

Apprende una rappresentazione dei dati mediante un conditional heterogeneous variational autoencoder. Cerca nello spazio latente e decodifica i candidati nello spazio originale, utilizzando le predizioni del classificatore per individuare un cambiamento di classe. È una baseline utile per confrontare la plausibilità ottenuta attraverso un modello generativo con quella ottenuta dalle regioni certificate di CertCF. [Paper originale](https://arxiv.org/abs/1910.09398)

### VCNet

**Titolo:** *VCNet: A Self-Explaining Model for Realistic Counterfactual Generation*.

**Anno:** 2022, ECML PKDD.

**Ambito:** modello autoesplicativo, studiato principalmente su dati tabulari.

Addestra congiuntamente un predittore e un generatore di counterfactual basato su un VAE condizionale. Dopo l'addestramento produce counterfactual senza risolvere da zero un problema di ottimizzazione per ogni query. Per confrontarlo correttamente con CertCF occorre però tenere conto che non è un semplice explainer post-hoc applicabile invariato al nostro classificatore già addestrato. [Paper originale](https://arxiv.org/abs/2212.10847)

## Altri candidati soprattutto recenti

### Regioni robuste di Maragno e colleghi

**Titolo:** *Finding Regions of Counterfactual Explanations via Robust Optimization*.

**Anno:** 2024, INFORMS Journal on Computing.

**Ambito:** formulazione generale per diversi classificatori, con applicazioni tabulari.

Usa ottimizzazione robusta per trovare counterfactual accompagnati da regioni che mantengono la predizione desiderata sotto perturbazioni limitate delle feature. Presenta procedure per diverse famiglie di modelli, comprese reti neurali e modelli ad alberi, sfruttando opportune formulazioni di ottimizzazione. È uno dei confronti concettualmente più vicini a CertCF, perché riguarda intere regioni valide, ma le costruisce attraverso una procedura legata alla singola query. [Paper originale](https://arxiv.org/abs/2301.11113)

### TABCF

**Titolo:** *TABCF: Counterfactual Explanations for Tabular Data Using a Transformer-Based VAE*.

**Anno:** 2024, ICAIF.

**Ambito:** specificamente tabulare, con feature numeriche e categoriche.

Utilizza un VAE basato su Transformer per cercare counterfactual in una rappresentazione latente dei dati. Introduce una ricostruzione differenziabile delle variabili categoriche mediante Gumbel-Softmax, affrontando anche gli sbilanciamenti che possono favorire modifiche di un tipo di feature rispetto a un altro. È un candidato moderno e direttamente pertinente ai nostri dataset, soprattutto per confrontare plausibilità e gestione delle variabili categoriche. [Paper e implementazione degli autori](https://github.com/Panagiotou/TABCF)

### FastDCFlow

**Titolo:** *Model-Based Counterfactual Explanations Incorporating Feature Space Attributes for Tabular Data*.

**Anno:** 2024, IJCNN.

**Ambito:** specificamente tabulare.

Utilizza normalizing flows per apprendere uno spazio latente che tenga conto della distribuzione dei dati, della prossimità e dell'obiettivo predittivo. Per le variabili categoriche impiega TargetEncoding e considera il costo delle modifiche nella rappresentazione delle feature. È particolarmente interessante per confrontare CertCF con un approccio generativo progettato per ridurre il costo della generazione ripetuta di counterfactual. [Paper originale](https://arxiv.org/abs/2404.13224)

### TDCE

**Titolo:** *Tabular Diffusion Counterfactual Explanations*.

**Anno:** 2026, Frontiers in Artificial Intelligence; preprint 2025.

**Ambito:** dati tabulari misti, numerici e categorici.

Genera counterfactual mediante un modello di diffusione, guidando il processo di denoising verso la classe desiderata e verso soluzioni vicine alla query. Gestisce variabili categoriche attraverso un'approssimazione differenziabile e permette di preservare feature immutabili. Aggiungerebbe al benchmark una famiglia generativa recente, ma richiede di considerare sia l'addestramento del generatore sia l'accesso al classificatore durante la generazione. [Paper originale](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1743495/full)

### DensityFlow

**Titolo:** *Density-Guided Robust Counterfactual Explanations on Tabular Data under Model Multiplicity*.

**Anno:** 2026, ICML.

**Ambito:** specificamente tabulare.

Modella la generazione del counterfactual come una dinamica continua mediante Neural ODE, guidata da una stima differenziabile della densità dei dati. Cerca di evitare regioni poco supportate, nelle quali classificatori diversi possono produrre predizioni discordanti, e include una variante black-box basata su un surrogato locale. È un candidato recente molto pertinente per il confronto sulla plausibilità, mentre la robustezza studiata riguarda la molteplicità dei modelli e non coincide con la certificazione locale di CertCF. [Paper originale](https://proceedings.mlr.press/v306/tan26a.html)

### APΔS

**Titolo:** *Probabilistically Robust Counterfactual Explanations under Model Changes*.

**Anno:** 2026, Artificial Intelligence.

**Ambito:** reti neurali di diverse architetture, con esperimenti tabulari anche su TabNet.

Introduce una procedura di certificazione probabilistica per stimare quanto un counterfactual tolleri variazioni dei parametri del modello. La procedura viene integrata nella generazione, verificando i candidati e proseguendo la ricerca quando non soddisfano il requisito di robustezza. Per il nostro benchmark andrebbe specificata anche la procedura di generazione associata ad APΔS, perché il contributo centrale è il test di robustezza e non un unico generatore indipendente. [Paper originale](https://www.sciencedirect.com/science/article/pii/S000437022500178X)

### CounterNet

**Titolo:** *CounterNet: End-to-End Training of Prediction Aware Counterfactual Explanations*.

**Anno:** 2023, KDD.

**Ambito:** principalmente dati tabulari; architettura con predittore ed explainer congiunti.

Addestra insieme un classificatore e una rete che produce counterfactual, combinando obiettivi di accuratezza predittiva, validità e prossimità. L'ottimizzazione viene organizzata per gestire i conflitti tra questi obiettivi e consentire una generazione rapida dopo l'addestramento. Come VCNet, sarebbe interessante per il confronto sulla latenza, ma richiede un protocollo che tenga conto dell'addestramento congiunto del classificatore. [Paper originale](https://arxiv.org/abs/2109.07557)

### COCOA

**Titolo:** *COCOA: Cost-Optimized COunterfactuAl explanation method*.

**Anno:** 2024, Information Sciences.

**Ambito:** dati tabulari e decisioni con costi dipendenti dall'istanza.

Cerca counterfactual tenendo conto dei costi associati alle decisioni, che possono variare da un individuo all'altro. Utilizza una formulazione cost-sensitive basata su un discriminante bayesiano e presenta un'implementazione con reti multilayer perceptron. È utile se vogliamo studiare esplicitamente costi decisionali o di recourse, ma è meno immediatamente confrontabile nel nostro benchmark basato prevalentemente su distanza e qualità del punto restituito. [Paper originale](https://www.sciencedirect.com/science/article/pii/S0020025524005292)

## Un candidato recente specifico per immagini

### LeapFactual

**Titolo:** *LeapFactual: Reliable Visual Counterfactual Explanation Using Conditional Flow Matching*.

**Anno:** 2025, NeurIPS.

**Ambito:** immagini e spiegazioni visive.

Utilizza conditional flow matching per generare immagini counterfactual che modificano l'esito del classificatore. Affronta difficoltà dei precedenti approcci generativi, tra cui gradienti poco informativi e disallineamenti tra lo spazio generativo e la frontiera decisionale. È interessante per un'eventuale estensione visiva di CertCF, ma non lo inserirei direttamente nel benchmark dei sette dataset tabulari. [Paper originale](https://proceedings.neurips.cc/paper_files/paper/2025/hash/ac1af21780d24a1345774c4a2e383972-Abstract-Conference.html)

## Selezione preliminare per i nuovi esperimenti

Le priorità seguenti riguardano le domande sperimentali che ciascun metodo permetterebbe di affrontare, non costituiscono un ranking delle prestazioni.

| Obiettivo del confronto | Candidati prioritari |
| --- | --- |
| Ricerca formale online rispetto a certificazione offline | MACE |
| Robustezza alle perturbazioni dell'input | PROBE, CROCO |
| Regioni interamente robuste | Metodo di Maragno e colleghi |
| Plausibilità attraverso generazione latente | C-CHVAE, TABCF |
| Famiglie generative più recenti | TDCE, DensityFlow |
| Efficienza della generazione dopo preparazione offline | FastDCFlow |
| Robustezza rispetto a cambiamenti del modello | PROPLACE, APΔS |

Per CertCF, una prima selezione è **MACE, CROCO o PROBE, il metodo di Maragno, TABCF e un metodo tra TDCE e DensityFlow**. Coprirebbero lacune sperimentali diverse, senza aggiungere molte baseline sostanzialmente ridondanti.

VCNet e CounterNet restano in un secondo gruppo, perché il confronto richiede maggiore attenzione al classificatore utilizzato. Le diverse nozioni di robustezza devono rimanere separate: è possibile misurare tutti i metodi con un protocollo comune, ma una garanzia sui cambiamenti del modello non equivale a una garanzia sulle perturbazioni dell'input.

Prima dell'integrazione occorre verificare l'eseguibilità delle implementazioni, le dipendenze e la compatibilità con il preprocessing del benchmark. Questi controlli non sono stati eseguiti nell'ambito della presente ricognizione bibliografica.
