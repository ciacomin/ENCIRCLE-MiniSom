### AGGIORNAMENTO 05-08-2024

#### Nuovi file temporanei. 
Piccolo disclaimer, io uso spyder quindi generalmente faccio girare solo le celle di codice a cui sono interessata, quindi se siete soliti girare l'intero codice probabilmente ci sono alcune celle vanno commentate di volta in volta. Sicuramente si potrebbe risolvere questa cosa mettendo due righe di codice aggiuntive e inserendo le variabili scelte dalla riga di comando, ma ancora non mi ci sono messa in modo approfondito. 

Per preparare i codici delle SOM utilizzare
- 1_6_SOM_dataset_prep
Nella prima parte si può scegliere quali estremi andare a considerare, nella seconda parte si sceglie quali variabili utilizzare nella SOM. IMPORTANTE: Salvarsi il valore della normalizzazione. 

Per creare le SOM da utilizzare
- 1_3_SOM_2_all_CAT_split_Z500_and_mSLP (per la SOM di Z500 e mSLP o Z500 e TCWV. Se si sceglie il secondo vanno commentate alcune cose e "scommentate" altre per scegliere la scala corretta dei grafici, ma è segnato. )
- 1_5_SOM_pr_generates (per la SOM di pr - prima parte del file - e per quella di Z500 e pr - seconda parte del file).

Per i composites ho cercato di generalizzare un po' utilizzando una struttura a 5 file
- functions_new_version
- functions_SOM
- SOM_data_file
- SOM_variable_file
- SOM_composites_file.
  
Ho notato che se mando il codice una volta e se in un secondo momento cambio le variabili nel file "SOM_variable_file" in realtà non si aggiornano i valori nel modo corretto, quindi fate attenzione a questa cosa.

Non sono i file definitivi perchè vorrei cercare di arrivare ad un punto in cui sono automatizzati e ottimizzati, però essendoci tutte queste scelte possibili tra dataset e variabili voglio anche essere sicura che il codice vada a pescare le cose giuste. 

Se non funziona qualcosa o se avete dubbi scrivetemi. 



