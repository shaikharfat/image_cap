
import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt
#from wordcloud import WordCloud
from copy import copy
from collections import Counter

def create_df_txt(text):
    """
    Create a data frame of the captions
    """
    datatxt = []
    for line in text.split('\n'):
        col = line.split(',')
        if len(col) == 1:
            continue
        w = col[0].split("#")
        datatxt.append(w + [col[1].lower()])

    df_txt = pd.DataFrame(datatxt,columns=["filename","index","caption"])
    return df_txt

    """
def remove_punctuation(text_original):
    Remove the punctuations from a single caption
    """
    if(isinstance(text_original,float)):
        print(text_original)
        text_original = 'abxdefghi'
    
    text_no_punctuation = text_original.translate(string.punctuation)
    return(text_no_punctuation)
    

def remove_single_character(text):
    """
    Remove the single character words from a single caption
    """
    text_len_more_than1 = ""
    for word in text.split():
        if len(word) > 1:
            text_len_more_than1 += " " + word
    return(text_len_more_than1)

def remove_numeric(text,printTF=False):
    """
    Remove words with numeric values from a single caption
    """
    text_no_numeric = ""
    for word in text.split():
        isalpha = word.isalpha()
        if printTF:
            print("    {:10} : {:}".format(word,isalpha))
        if isalpha:
            text_no_numeric += " " + word
    return(text_no_numeric)

def text_clean(text_original):
    """
    Clean a single caption by removing the punctuations, the single character words and the words with numeric values
    """
    text = remove_punctuation(text_original)
    text = remove_single_character(text)
    text = remove_numeric(text)
    return(text)

def add_start_end_seq_token(captions):
    """
    Add start and end sequence tokens of all the captions
    """
    caps = []
    for txt in captions:
        txt = 'startseq ' + txt + ' endseq'
        caps.append(txt)
    return(caps)

def generate_word_cloud(df_txt):
    """
    Generate a word cloud of all the captions
    """
    wordcloud = WordCloud(max_font_size=60).generate(df_txt)
    plt.figure(figsize = (16, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig('C:/Users/Xpro/autocaption/image/wordCloud_raw2.png', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':

    
    #df_txt = pd.read_csv(filepath_or_buffer = r"C:\Users\Xpro\Flickr30k\data\results.csv",sep = '|')
    #df_txt[' comment_number']=0
    #df_txt.to_csv('C:/Users/Xpro/Flickr30k/data/newdata.txt', sep=',', index=False)
   #print(df_txt.head())
    #dir_Flickr_text = "C:/Users/Xpro/Flickr30k/data/newdata.txt"

    # Read in the Flickr caption data
    #file = open(dir_Flickr_text,'r',encoding="utf8")
    #text = file.read()
    #file.close()

    #df_txt = create_df_txt(text) 
    #df_txt=pd.read_csv(dir_Flickr_text,sep=',')
    #df_txt.columns = ["filename","index","caption"]
    #image_name| comment_number| comment
    
    #uni_filenames = np.unique(df_txt.filename.values)
    #print("The number of unique file names : {}".format(len(uni_filenames)))
    #print("The distribution of the number of captions for each image:")
    #print(Counter(Counter(df_txt.filename.values).values()))

    # Clean captions
    #print(df_txt.columns)
    #Index(['image_name', ' comment_number', ' comment'], dtype='object')
    #df_txt.rename(columns={' comment_number': 'index', ' comment':'caption'},inplace=True)
    #print(df_txt.columns)
    #for i, caption in enumerate(df_txt.caption.values):
        #newcaption = text_clean(caption)
        #df_txt["caption"].iloc[i] = newcaption
    #print(df_txt.caption.head())
        
    # Add start and end sequence tokens
    #df_txt0 = copy(df_txt)
    #df_txt0["caption"] = add_start_end_seq_token(df_txt["caption"])
    
    #del df_txt
    df_txt0= pd.read_csv('C:/Users/Xpro/Flickr30k/data/cleandata.txt', sep='\t')
    df_token0 = copy(df_txt0)
    
    for i in range(0,20):
        df_token0['index'] = df_token0.append({'index': i}, ignore_index=True)
    print(df_token0['index'].head())
    
    
    df_token0 = df_token0.loc[df_token0["index"].values %5 == 0,: ] #because boolean
    print(df_token0.head(10))
    
    df_token0['index']=0
    df_token0.to_csv('C:/Users/Xpro/Flickr30k/data/token0.txt', sep=',', index=False)    

    '''for i in range (0,20):
        if(df_token0['index'])
    
    #df_token1 = pd.DataFrame(columns=["filename","index","caption"])
    #print(type(df_token1))
    #df_token1.columns = ["filename","index","caption"]
    df_token1=df_txt0.iloc[0]
    
    for i in range(1,20):
        #print(df_txt0.iloc[i])
        if (i%5==0):
            df_token0=df_txt0.iloc[i]
            print(df_token0)
            df_token0=df_token0.to_frame()
            print(type(df_token0))
            df_token1=df_token1.append(df_token0)
    
    #df_token1.columns = ["filename","index","caption"]
    df_token1=df_token1.to_frame()
    #df_token1.columns = ["filename","index","caption"]
    print(type(df_token1))

    #df_txt0 = df_txt0.loc[df_txt0["index"].values == "0",: ] #because boolean
    df_token1.to_csv('C:/Users/Xpro/Flickr30k/data/token0.txt', sep=',', index=False)
    df_txt0_new = pd.read_csv('C:/Users/Xpro/Flickr30k/data/token0.txt', sep='\n')
    df_txt0_new.columns = ["filename","index","caption"]
    print(df_txt0_new)
    #df_txt0_new.describe()
    #generate_word_cloud(df_txt0)'''