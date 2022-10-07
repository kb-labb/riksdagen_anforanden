mkdir data
mkdir data/json

wget https://data.riksdagen.se/dataset/anforande/anforande-202122.json.zip -P data/json
wget https://data.riksdagen.se/dataset/anforande/anforande-202021.json.zip -P data/json
wget https://data.riksdagen.se/dataset/anforande/anforande-201920.json.zip -P data/json
wget https://data.riksdagen.se/dataset/anforande/anforande-201819.json.zip -P data/json
wget https://data.riksdagen.se/dataset/anforande/anforande-201718.json.zip -P data/json
wget https://data.riksdagen.se/dataset/anforande/anforande-201617.json.zip -P data/json
wget https://data.riksdagen.se/dataset/anforande/anforande-201516.json.zip -P data/json
wget https://data.riksdagen.se/dataset/anforande/anforande-201415.json.zip -P data/json

unzip data_json/anforande-202122.json.zip -d data/json
unzip data_json/anforande-202021.json.zip -d data/json
unzip data_json/anforande-201920.json.zip -d data/json
unzip data_json/anforande-201819.json.zip -d data/json
unzip data_json/anforande-201718.json.zip -d data/json
unzip data_json/anforande-201617.json.zip -d data/json
unzip data_json/anforande-201516.json.zip -d data/json
unzip data_json/anforande-201415.json.zip -d data/json

rm data/json/*.json.zip