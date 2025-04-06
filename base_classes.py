class Card():
    def __init__(self, char):
        self.char = char
        self.value = self.getValue()
    
    def getValue(self):
        if(self.char.isdigit()):
            return int(self.char)
        else:
            return 10
    
class Ace(Card):
    def __init__(self, char='A'):
        super().__init__(char)
        self.value = 11

    def switchValue11(self):    
        self.value = 11

    def switchValue1(self):
        self.value = 1

class Deck():
    def __init__(self):
        self.cards = []
        for i in range(13):
            for j in range(4):
                if 1 <= i <= 9:
                    card = Card(str(i + 1))
                elif i == 0:
                    card = Ace()
                elif i == 10:
                    card = Card('J')
                elif i == 11:
                    card = Card('Q')
                elif i == 12:
                    card = Card('K')
                self.cards.append(card)
    
    def show(self):
        for idx, card in enumerate(self.cards):
            print(f"{idx+1}: {card.char}, value = {card.value}")
