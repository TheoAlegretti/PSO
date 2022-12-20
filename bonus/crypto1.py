#fonction d'encryptage 1 en utilisant la suite des nombre premier (test insta )

def encryp1() : 
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
    Min = input('Votre message a encrypter ? : ')
    #on mets tous en lower dans un premier temps 
    crypted = ''
    for letter in range(0,len(Min),1) : 
        if Min[letter] == " " : 
            crypted = crypted + '.'
        elif Min[letter] == '.' : 
            crypted = crypted + '!'
        elif Min[letter] == "\'" : 
            crypted = crypted + '/'
        elif Min[letter] == "," : 
            crypted = crypted + '/'
        elif Min[letter] == ":" : 
            crypted = crypted + ':'
        elif Min[letter] == "!" : 
            crypted = crypted + '!'
        else : 
            caract = unidecode.unidecode(str(Min[letter].lower()))
            if letter == 0 : 
                crypted = crypted  + str(Key[caract])
            else : 
                crypted = crypted  + str(Key[caract]) #pour aider on peut séparer chaque lettre 
    print(f'Your crypted message : {crypted}' )
            
encryp1()