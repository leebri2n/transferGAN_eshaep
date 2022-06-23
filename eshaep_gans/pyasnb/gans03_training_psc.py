from tqdm import tqdm

s = 'sadlkjfa;sldkja;sdklfjas;dklfja;kfajd;kflja;ldkfajsdk;fajf'

pbar = tqdm(s)
for char in s:
    print(char)
    pbar.update()
