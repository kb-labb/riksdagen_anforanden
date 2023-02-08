mkdir data
mkdir data/json
mkdir data/audio

wget https://data.riksdagen.se/dataset/anforande/anforande-202223.json.zip -P data/json
wget https://data.riksdagen.se/dataset/anforande/anforande-202122.json.zip -P data/json
wget https://data.riksdagen.se/dataset/anforande/anforande-202021.json.zip -P data/json
wget https://data.riksdagen.se/dataset/anforande/anforande-201920.json.zip -P data/json
wget https://data.riksdagen.se/dataset/anforande/anforande-201819.json.zip -P data/json
wget https://data.riksdagen.se/dataset/anforande/anforande-201718.json.zip -P data/json
wget https://data.riksdagen.se/dataset/anforande/anforande-201617.json.zip -P data/json
wget https://data.riksdagen.se/dataset/anforande/anforande-201516.json.zip -P data/json
wget https://data.riksdagen.se/dataset/anforande/anforande-201415.json.zip -P data/json
wget https://data.riksdagen.se/dataset/anforande/anforande-201314.json.zip -P data/json
wget https://data.riksdagen.se/dataset/anforande/anforande-201213.json.zip -P data/json
wget https://data.riksdagen.se/dataset/anforande/anforande-201112.json.zip -P data/json
wget https://data.riksdagen.se/dataset/anforande/anforande-201011.json.zip -P data/json
wget https://data.riksdagen.se/dataset/anforande/anforande-200910.json.zip -P data/json
wget https://data.riksdagen.se/dataset/anforande/anforande-200809.json.zip -P data/json
wget https://data.riksdagen.se/dataset/anforande/anforande-200708.json.zip -P data/json
wget https://data.riksdagen.se/dataset/anforande/anforande-200607.json.zip -P data/json
wget https://data.riksdagen.se/dataset/anforande/anforande-200506.json.zip -P data/json
wget https://data.riksdagen.se/dataset/anforande/anforande-200405.json.zip -P data/json
wget https://data.riksdagen.se/dataset/anforande/anforande-200304.json.zip -P data/json
wget https://data.riksdagen.se/dataset/anforande/anforande-200203.json.zip -P data/json
wget https://data.riksdagen.se/dataset/anforande/anforande-200102.json.zip -P data/json
wget https://data.riksdagen.se/dataset/anforande/anforande-200001.json.zip -P data/json
wget https://data.riksdagen.se/dataset/anforande/anforande-199900.json.zip -P data/json
wget https://data.riksdagen.se/dataset/anforande/anforande-199899.json.zip -P data/json
wget https://data.riksdagen.se/dataset/anforande/anforande-199798.json.zip -P data/json
wget https://data.riksdagen.se/dataset/anforande/anforande-199697.json.zip -P data/json
wget https://data.riksdagen.se/dataset/anforande/anforande-199596.json.zip -P data/json
wget https://data.riksdagen.se/dataset/anforande/anforande-199495.json.zip -P data/json
wget https://data.riksdagen.se/dataset/anforande/anforande-199394.json.zip -P data/json

unzip data/json/anforande-202223.json.zip -d data/json
unzip data/json/anforande-202122.json.zip -d data/json
unzip data/json/anforande-202021.json.zip -d data/json
unzip data/json/anforande-201920.json.zip -d data/json
unzip data/json/anforande-201819.json.zip -d data/json
unzip data/json/anforande-201718.json.zip -d data/json
unzip data/json/anforande-201617.json.zip -d data/json
unzip data/json/anforande-201516.json.zip -d data/json
unzip data/json/anforande-201415.json.zip -d data/json
unzip data/json/anforande-201314.json.zip -d data/json
unzip data/json/anforande-201213.json.zip -d data/json
unzip data/json/anforande-201112.json.zip -d data/json
unzip data/json/anforande-201011.json.zip -d data/json
unzip data/json/anforande-200910.json.zip -d data/json
unzip data/json/anforande-200809.json.zip -d data/json
unzip data/json/anforande-200708.json.zip -d data/json
unzip data/json/anforande-200607.json.zip -d data/json
unzip data/json/anforande-200506.json.zip -d data/json
unzip data/json/anforande-200405.json.zip -d data/json
unzip data/json/anforande-200304.json.zip -d data/json
unzip data/json/anforande-200203.json.zip -d data/json
unzip data/json/anforande-200102.json.zip -d data/json
unzip data/json/anforande-200001.json.zip -d data/json
unzip data/json/anforande-199900.json.zip -d data/json
unzip data/json/anforande-199899.json.zip -d data/json
unzip data/json/anforande-199798.json.zip -d data/json
unzip data/json/anforande-199697.json.zip -d data/json
unzip data/json/anforande-199596.json.zip -d data/json
unzip data/json/anforande-199495.json.zip -d data/json
unzip data/json/anforande-199394.json.zip -d data/json

rm data/json/*.json.zip