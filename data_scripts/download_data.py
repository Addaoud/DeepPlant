import requests
from bs4 import BeautifulSoup
import time
import os
import argparse
data_path="/s/chromatin/b/nobackup/deepplant/Data/"
base_url = "https://biobigdata.nju.edu.cn/ChIPHub_download/"
parser = argparse.ArgumentParser(description='Data Download')
parser.add_argument('--species-name', type=str, default="arabidopsis_thaliana", metavar='N',
                        help='Download data from Chiphub i.e. either oryza_sativa or aradopsis_thaliana')
parser.add_argument('--data-path', type=str, default="/s/chromatin/b/nobackup/deepplant/Data/", metavar='N',
                        help='Place to store downloaded data')
args = parser.parse_args()
final_url=base_url+args.species_name+"/"
response = requests.get(final_url)
final_data_path=args.data_path+args.species_name+"/Chiphub_Final/"
response.raise_for_status()

os.makedirs(final_data_path,exist_ok=True)
soup = BeautifulSoup(response.content, 'html.parser')
experiment_links = [link.get('href') for link in soup.find_all('a')]
for exp_link in experiment_links[2:-1]:
    experiment_name = exp_link.strip('/')
    signal_url = f"{final_url}{experiment_name}/signal/"
    time.sleep(50)
    download_experiment_path=f"{final_data_path}/{experiment_name}/signal"
    os.makedirs(download_experiment_path,exist_ok=True)
    print(f"Downloading files from {signal_url} ...")
    try:
        signal_response = requests.get(signal_url,timeout=10)
        signal_soup = BeautifulSoup(signal_response.content, 'html.parser')
        
        file_links = [link.get('href') for link in signal_soup.find_all('a') if "final" in link.get('href', '') and  link.get('href').endswith('rpgc.bw')]  # assuming files are .gz, modify as per requirement

        for file_link in file_links:
            time.sleep(30)
            file_url = f"{signal_url}{file_link}"
            try:
                file_content = requests.get(file_url,timeout=10).content
                file_path = f"{download_experiment_path}/{file_link}"     
                print(file_path)
                with open(file_path, 'wb') as f:
                    f.write(file_content)
            except:
                pass
    except:
        pass
print("Download complete!")

