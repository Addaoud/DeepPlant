import subprocess
import tempfile
import pandas as pd
import numpy as np
import os
def create_blast_db(fasta_file, db_name="test", db_type='nucl'):
    """
    Create a BLAST database from a FASTA file.
    """
    cmd = ['makeblastdb', '-in', fasta_file, '-dbtype', db_type, '-out', db_name]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def run_blast(df_modified,seq, database,chromosome): 

    similarity_values = []
    width=[]
    chr1=[]
    chr2=[]
    seq_idx=[]
    sstart=[]
    send=[]
    start_csv=[]
    end_csv=[]

    qstart=[]
    qend=[]
    con_com=[]
    for i1 in range(len(df_modified)):
        sequence=seq[df_modified["start_pos"][i1]:df_modified["end_pos"][i1]]     
        print(i1)
        # Create a temporary file for the input sequence
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_fasta:
            temp_fasta.write(f">{chromosome}\n{sequence}\n")
            temp_fasta_path1 = temp_fasta.name
            # Construct the BLAST command
        evalue = 1e-5
        blast_command = ['blastn', '-query', temp_fasta_path1, '-db', database, '-outfmt', '6', '-num_threads', '18','-evalue', str(evalue)]

            # Execute the BLAST command and capture the output
        result = subprocess.run(blast_command, capture_output=True, text=True)

            # Process the BLAST output to extract similarity values
        for line in result.stdout.strip().split('\n'):
                columns = line.split('\t')
                if len(columns) > 2:# and columns[1]!=chromosome:  # Ensure there's a percentage identity column
                    similarity_values.append(float(columns[2]))  # Column 3 is the percentage identity
                    width.append(float(columns[3]))
                    chr1.append(columns[0])
                    chr2.append(columns[1])
                    qstart.append(float(columns[6]))
                    qend.append(float(columns[7]))
                    sstart.append(float(columns[8]))
                    send.append(float(columns[9]))
                    seq_idx.append(df_modified["seq_idx"][i1])
                    start_csv.append(df_modified["start_pos"][i1])
                    end_csv.append(df_modified["end_pos"][i1])
        # Optionally, remove the temporary file
        subprocess.run(['rm', temp_fasta_path1])
    #return con_com
    return seq_idx,chr1,chr2,width,similarity_values,qstart,qend,start_csv,end_csv,sstart,send
    
# Example usage
def read_fasta(fasta_file):
    with open(fasta_file, 'r') as file:
        sequence = ''
        for line in file:
            if line.startswith('>'):
                header = line  # Reset header for the new sequence
            else:
                sequence += line  # Add lines of the sequence
    return sequence
# Get the input location from the arguments
#species="Arabidopsis thaliana"
species="Oryza Sativa"
mask="masked"
size=10
filename = f"/s/chromatin/b/nobackup/deepplant/Data/genomes_chunked_{size}kb_1150_{mask}.csv"
df=pd.read_csv(filename)
df1=(df[df["genome"]==species]).reset_index()
genome=np.array(df1["genome"])
sequence_idx=np.array(df1["seq_idx"])
start_pos=np.array(df1["start_pos"])
end_pos=np.array(df1["end_pos"])
chromosome=np.array(df1["chromosome"])
if species=="Arabidopsis thaliana":
    list1=["chromosome 1","chromosome 2","chromosome 3","chromosome 4","chromosome 5"]
    database = f"/s/chromatin/b/nobackup/deepplant/Data/arabidopsis_thaliana/blast_db/arabidopsis_{mask}"  #The BLAST database you created
    output_dir=f"/s/chromatin/b/nobackup/deepplant/Data/arabidopsis_thaliana/blast_output/{mask}_{species}_{size}_kb_overlap"
    temp_file=f"/s/chromatin/b/nobackup/deepplant/Data/arabidopsis_thaliana/genome/"
    temp_species="arabidopsis_thaliana"
else:
    list1=["chromosome 1","chromosome 2","chromosome 3","chromosome 4","chromosome 5","chromosome 6","chromosome 7","chromosome 8","chromosome 9","chromosome 10","chromosome 11","chromosome 12"]
    database = f"/s/chromatin/b/nobackup/deepplant/Data/oryza_sativa/blast_db/oryza_{mask}"  #The BLAST database you created
    output_dir=f"/s/chromatin/b/nobackup/deepplant/Data/oryza_sativa/blast_output/{mask}_{species}_{size}kb_overlap"
    temp_file=f"/s/chromatin/b/nobackup/deepplant/Data/oryza_sativa/genome/"
    temp_species="oryza_sativa"
os.makedirs(output_dir,exist_ok=True)
for i in range(len(list1)):
    file1=f'{temp_file}/Chr{list1[i].split(" ")[-1]}_{mask}_{temp_species}.fasta'
    seq=read_fasta(file1) 
    df_modified=(df1[df1["chromosome"]==list1[i]]).reset_index()
    seq_idx,chr1,chr2,width,similarity_values,qstart,qend,start_csv,end_csv,sstart,send = run_blast(df_modified,seq,database,f'Chr{list1[i].split(" ")[-1]}')
            #con_com=run_blast(df_modified,df_modified2,seq,database,f'Chr{list1[i].split(" ")[-1]}',seq2,f'Chr{list1[j].split(" ")[-1]}')
            #con_com=np.array(-1,2) 
            #np.save(f"Connected_Component_Chromosome_{list1[i].split(' ')[-1]}_Chromosome_{list1[j].split(' ')[-1]}.npy",con_com)
    df2=pd.DataFrame({"seq_idx":seq_idx,"chr1":chr1,"chr2":chr2,"width":width,"similarity":similarity_values,"qstart":qstart,"qend":qend,"start_csv":start_csv,"end_csv":end_csv,"sstart":sstart,"send":send})
    df2.to_csv(f"{output_dir}_{list1[i].split(' ')[-1]}.csv")
