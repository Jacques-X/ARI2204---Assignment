import random

class Card():
    def __init__(self, char):
        self.char = char
        self.value = self.assignValue()

    def assignValue(self):
        # Assigns numerical value or 10 for face cards
        if(self.char.isdigit()):
            return int(self.char)
        else:
            return 10

    def getValue(self):
        return self.value

class Ace(Card):
    def __init__(self, char='A'):
        super().__init__(char)
        # Initial value for Ace is 11
        self.value = 11

    def switchValue11(self):
        self.value = 11

    def switchValue1(self):
        self.value = 1

class Deck():
    def __init__(self):
        self.cards = []
        # Creates a standard 52-card deck
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
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.cards)

    def takeTopCard(self):
        # Removes and returns the top card from the deck
        return self.cards.pop(0)

    def show(self):
        for idx, card in enumerate(self.cards):
            print(f"{idx+1}: {card.char}, value = {card.value}")

class Player():
    def __init__(self, deck):
        # Deals initial two cards
        self.cards = [deck.takeTopCard(), deck.takeTopCard()]
        self.stands = False

    def hit(self, deck):
        # Adds a card if the player hasn't stood
        if not self.stands:
            self.cards.append(deck.takeTopCard())
        else:
            print("Can't hit after choosing to stand")

    def getValue(self):
        # Calculates the total value of the hand
        return sum(card.getValue() for card in self.cards)

    def isBusted(self):
        # Checks if the hand value exceeds 21
        return self.getValue() > 21

    def showHand(self):
        return [(card.char, card.value) for card in self.cards]

class Dealer(Player):
    def __init__(self, deck):
        # Deals initial card (typically one face up)
        self.cards = [deck.takeTopCard()]
        self.stands = False

    def playTurn(self, deck):
        # Dealer hits until value is 17 or more
        while self.getValue() < 17:
            self.hit(deck)

class BlackJackEnv():
    def __init__(self):
        self.deck = Deck()
        self.player = Player(self.deck)
        self.dealer = Dealer(self.deck)

    def get_state(self):
        # Returns game state: player sum, dealer's up card value, usable ace flag
        player_sum = self.player.getValue()
        dealer_card = self.dealer.cards[0].getValue()
        usable_ace = any(card.char == 'A' and card.value == 11 for card in self.player.cards)
        return (player_sum, dealer_card, usable_ace)

    def step(self, action):
        # Performs an action (hit or stand) in the environment
        if action == "hit":
            self.player.hit(self.deck)
            if self.player.isBusted():
                # Returns state, reward (-1), done (True)
                return self.get_state(), -1, True
            else:
                # Returns state, reward (0), done (False)
                return self.get_state(), 0, False
        elif action == "stand":
            self.dealer.playTurn(self.deck)
            reward = self.calculate_reward()
            # Returns final state, reward, done (True)
            return self.get_state(), reward, True

    def playRound(self):
        # Plays a round of Blackjack with user interaction
        print("Your hand:", self.player.showHand())
        # Shows dealer's initial hand (typically one card is face down)
        print("Dealer hand:", self.dealer.showHand())
        # Player's turn
        while not self.player.stands and not self.player.isBusted():
            choice = input("Hit or Stand? ").lower()
            if choice == 'hit':
                self.player.hit(self.deck)
                print("You drew:", self.player.cards[-1].char)
                print("Your hand:", self.player.showHand(), "Value:", self.player.getValue())
            elif choice == 'stand':
                self.player.stands = True
            else:
                print("Please type 'hit' or 'stand'.")

        if self.player.isBusted():
            print("You busted! Dealer wins.")
            return

        print("Dealer's turn...")
        self.dealer.playTurn(self.deck)
        print("Dealer's hand:", self.dealer.showHand(), "Value:", self.dealer.getValue())

        result = self.checkWinCondition()
        print(result)

    def checkWinCondition(self):
        # Determines the winner based on hand values
        player_val = self.player.getValue()
        dealer_val = self.dealer.getValue()

        if self.dealer.isBusted():
            return "Dealer busted. Player wins!"
        elif player_val > dealer_val:
            return "Player wins!"
        elif dealer_val > player_val:
            return "Dealer wins!"
        else:
            return "It's a tie!"

    def calculate_reward(self):
        # Calculates reward based on the game outcome
        if self.player.isBusted():
            return -1
        elif self.dealer.isBusted():
            return 1
        elif self.player.getValue() > self.dealer.getValue():
            return 1
        elif self.player.getValue() < self.dealer.getValue():
            return -1
        else:
            return 0

    def reset(self):
        # Resets the environment for a new round
        self.__init__()
        return self.get_state()