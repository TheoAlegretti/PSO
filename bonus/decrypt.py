def dencryp1() : 
    import string 
    import unidecode
    global Key,Min,crypted
    #26 premiers nombre premiers  : 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101
    nbp = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101]
    for num in range (0,len(nbp),1) : 
        nbp[num] += 1 #on applique le chiffre de césar pour décaler les valeurs des lettres 
    alphabet_string = string.ascii_lowercase
    alphabet_list = list(alphabet_string)
    #on contruit un dictionnaire => + rapide 
    Key = dict(zip(alphabet_string, nbp))
    Min = str(input('Your crypted message ? : '))
    def get_key(val):
        for key, value in Key.items():
            if val == value:
                return key
    #on mets tous en lower dans un premier temps 
    crypted = ''
    for letter in range(0,len(Min),1) : 
        if Min[letter] == "." : 
            crypted = crypted + ' '
            Min.replace(Min[letter],"")
        elif Min[letter] == "/" : 
            crypted = crypted + ' '
            Min.replace(Min[letter],"")
        elif Min[letter] == "!" : 
            crypted = crypted + '!'
            Min.replace(Min[letter],"")
        else : 
            a  = Min[letter:letter+3]
            b  = Min[letter:letter+2]
            c  = Min[letter]
            if ("." in a) or ("/" in a) : 
                if ("." in b) or ("/" in b) : 
                    if ("." in c) or ("/" in c) : 
                        continue 
                    else : 
                        C = int(Min[letter])
                else : 
                    if int(Min[letter:letter+2]) in list(Key.values()) : 
                        C = int(Min[letter:letter+2])
                    else : 
                        C = int(Min[letter])
            else : 
                if int(Min[letter:letter+3]) in list(Key.values()) : 
                    C = int(Min[letter:letter+3])
                elif int(Min[letter:letter+2]) in list(Key.values()): 
                    C = int(Min[letter:letter+2])
                else : 
                    C = int(Min[letter])
        crypted = crypted  + str(get_key(C))
    print(f'Your uncrypted message : {crypted}')
            
dencryp1()