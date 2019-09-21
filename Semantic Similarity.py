import os, time, math
os.chdir("/Users/chinki/Desktop/")

def cosine_similarity(vec1, vec2):
    dot_product, sum1, sum2 = 0,0,0
    for e in vec1:
        sum1 += vec1[e]**2
    for f in vec2:
        sum2 += vec2[f]**2
    for word in vec2:
        if word in vec1:
            dot_product += (vec1[word])*(vec2[word])
    if (sum1==0) or (sum2==0):
        return -1
    else:
        return dot_product/((math.sqrt(sum1))*(math.sqrt(sum2)))
def build_semantic_descriptors(L):
    dict = {}
    # examine the list L for each sentence. From each sentence, gather the group of words that are non-repeititive
    # add to the dictionary iff it's not in the dictionary
    for i in range(len(L)):
        for word in set(L[i]):
            if word not in dict:
                dict[word] = {}
            for j in set(L[i]):
                    if (j is not word) and (j in dict[word]):
                        dict[word][j] += 1
                    if (j is not word) and (j not in dict[word]):
                        dict[word][j] = 1
    return dict
def build_semantic_descriptors_from_files(filenames):
    t = ""
    list = []
    for i in range(len(filenames)):
        text = open(filenames[i], "r", encoding="utf-8")
        t += text.read()
    text = t.lower()
    text = text.replace("?",".")
    text = text.replace("!",".")
    text = text.split(".")
    for sentence in text:
        sentence = sentence.replace("--", " ")
        sentence = sentence.replace("-"," ")
        sentence = sentence.replace(":"," ")
        sentence = sentence.replace(";", " ")
        sentence = sentence.replace('"', " ")
        sentence = sentence.replace("'"," ")
        sentence = sentence.replace(",", " ")
        sentence = sentence.replace("\n", " ")
        sentence = sentence.split()
        list.append(sentence)
    giant_dict = build_semantic_descriptors(list)
    return giant_dict
def most_similar_word(word, choices, semantic_descriptors, similarity_fn):
    final_list = []
    max = -2
    right_word = ""
    for choice in choices:
        if choice not in semantic_descriptors or word not in semantic_descriptors:
            sim_value = -1
        else:
            sim_value = similarity_fn(semantic_descriptors[word],semantic_descriptors[choice])
        if sim_value > max:
            right_word = choice
            max = sim_value
    return right_word 
def run_similarity_test(filename, semantic_descriptors, similarity_fn):
    '''
    text = (open(filename, "r", encoding="utf-8")).read()'''
    count, total = 0,0
    list, list1 = [], []
    text = open(filename, "r", encoding="utf-8").read()
    text = text.split("\n")
    for sentence in text:
        list = sentence.split(" ")
        list1.append(list)
   
    for sentence in list1:
        if len(sentence) >=3 :
            right_choice = most_similar_word(sentence[0],sentence[2:],semantic_descriptors,similarity_fn)
            if right_choice == sentence[1]:
                count += 1
            total += 1
    return (count*100)/total
def simeuc(v1, v2):
    sum = 0
    for item in v1:
        if item in v2:
            sum += (v1[item]-v2[item])**2
        else:
            sum += (v1[item]-0)**2
        return -1*math.sqrt(sum)
def simeucnorm(v1,v2):
    v1sum, v2sum = 0,0
    v1total, v2total = 0,0
    sum = 0
    for i in v1:
        v1sum += v1[i]
        v1total += (v1[i])**2
        for i in v2:
            v2sum += v2[i]
            v2total += (v2[i])**2
            sum += ((v1sum/math.sqrt(v1total))-(v2sum/math.sqrt(v2total)))**2
    return -1*math.sqrt(sum)
# t1 = time.time()
#sd = (build_semantic_descriptors_from_files(["swannsway.txt"]))
#t2 = time.time() - t1
#print(t2)
#dict = build_semantic_descriptors(L)
#print(dict["man"])
#print(simeucnorm({'a':1, 'b':2,'c':3}, {'b':4, 'c':5, 'd':6}))
#print(simeuc({'a':1, 'b':2,'c':3}, {'b':4, 'c':5, 'd':6}))
#print(run_similarity_test("test.txt",build_semantic_descriptors_from_files(["swannsway.txt", "warandpeace.txt"]),simeuc      ))
#print(run_similarity_test("test.txt",build_semantic_descriptors_from_files(["swannsway.txt", "warandpeace.txt"]),simeucnorm      ))

def some_fn(filenames, n):
    t = ""
    L = []
    for i in range(len(filenames)):
        text = open(filenames[i], "r", encoding="utf-8")
        t += text.read()
    text = t.lower()
    text = text.replace("?",".")
    text = text.replace("!",".")
    text = text.split(".")
    for sentence in text:
        sentence = sentence.replace("--", " ")
        sentence = sentence.replace("-"," ")
        sentence = sentence.replace(":"," ")
        sentence = sentence.replace(";", " ")
        sentence = sentence.replace('"', " ")
        sentence = sentence.replace("'"," ")
        sentence = sentence.replace(",", " ")
        sentence = sentence.replace("\n", " ")
        sentence = sentence.split()
        L.append(sentence)
    #
    dict = {}
    # examine the list L for each sentence. From each sentence, gather the group of words that are non-repeititive
    # add to the dictionary iff it's not in the dictionary
    for i in range(int((len(L)/10)*n)):
        for word in set(L[i]):
            if word not in dict:
                dict[word] = {}
            for j in set(L[i]):
                    if (j is not word) and (j in dict[word]):
                        dict[word][j] += 1
                    if (j is not word) and (j not in dict[word]):
                        dict[word][j] = 1
    return dict
def tester():
    L = [["i", "am", "a", "sick", "man"],
    ["i", "am", "a", "spiteful", "man"],
    ["i", "am", "an", "unattractive", "man"],
    ["i", "believe", "my", "liver", "is", "diseased"],
    ["however", "i", "know", "nothing", "at", "all", "about", "my",
    "disease", "and", "do", "not", "know", "for", "certain", "what", "ails", "me"]]

    list = []
    for k in range(1,10):
        #some_fn(["swannsway.txt", "warandpeace.txt"], n)
        t1 = time.time()
        run_similarity_test("test.txt",some_fn(["swannsway.txt", "warandpeace.txt"], k),cosine_similarity)
        t2= time.time() - t1   
        list.append(t2)
    print(list)

tester()