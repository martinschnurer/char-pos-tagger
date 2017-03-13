
def getCharDict():
    i = 1
    dictionary = {}
    file = open('definedChars/chars')
    line = file.readline()[:-1]
    for char in line:
        dictionary[char] = i
        i = i + 1
    return dictionary



class CharDictionary:
    def __init__(self):
        self.dictionary = getCharDict()
        self.len = len(self.dictionary) + 1 # additional position for undefined char

    def getEmbedding(self, char):
        if char in self.dictionary:
            return self.dictionary[char]
        else:
            return 0
    def getVector(self, char):
        vec = [0] * self.len
        charPosition = self.getEmbedding(char)
        vec[charPosition] = 1
        return vec


if __name__ == '__main__':
    charDict = CharDictionary()
    vec = charDict.getVector('9')
    print(vec)
