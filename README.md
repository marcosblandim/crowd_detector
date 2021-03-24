# crowd_detector
- captura frame de camera local ou remota, a cada n segundos
- faz a detecção de aglomerações nesse frame e o guarda em uma pasta
- variáveis
    - fonte do frame
    - número de frames
    - intervalo entre frames
    - nove do arquivo
    - nome da pasta

## Requirements
- python 3.8^
- pip 20.2.3^
- virtualenv 20.0.31

## Usage
1. clone the repository and enter it
```shell
git clone https://github.com/uiot/crowd_detector.git
cd crowd_detector
```
2. create a virtual environment and activate it
```shell
virtualenv venv
source venv/bin/activate # Linux
.\venv\Scripts\activate # Windows
```
3. pip install requirements
```shell
pip install -r requirements.txt
```
4. download YOLO's weights to `yolo_files` folder
```shell
wget -O ./crowd_detector/yolo_detection/yolo_files/yolo.weights https://pjreddie.com/media/files/yolov3.weights
```
> you can configure the software by modifing the file ./crowd_detector/settings.yaml

## DOING
- melhorar/refazer/refatorar o código de detecção de aglomeração atual (fase 1)

## TODO
- salvar também o relatório de cada frame (?)
- começar fase 2
- pensar na fase 3
- implementar a captura de frames de dentro de um diretorio