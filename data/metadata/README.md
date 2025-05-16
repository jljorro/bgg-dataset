# Board Game Dataset Repository

This folder contains raw metadata and contextual annotations for board games. The data is stored in tab-separated values (TSV) format across several files. Each file captures different aspects of the games, including basic metadata, tags, descriptions, rankings, and contextual user-generated information.

---

## File Descriptions

### 1. raw_game_metadata_1.tsv

Contains general metadata for each game, including playtime, age recommendations, ratings, and popularity statistics.

**Columns:**
- id: Unique game identifier
- name: Game title
- url: Relative URL of the game on the data source
- year: Year of publication
- weight: Complexity rating (1–5 scale)
- age: Minimum recommended age
- minPlayTime, maxPlayTime: Minimum and maximum duration (in minutes)
- minNumberPlayers, maxNumberPlayers: Min and max number of players
- bestMinNumberPlayers, bestMaxNumberPlayers: Community-recommended player counts
- avgRating: Average user rating
- numRatings: Number of ratings
- numComments: Number of user comments
- numFans: Number of users who marked the game as a favorite

**Example:**
id    name        url             year  weight  age  minPlayTime  maxPlayTime  minNumberPlayers  maxNumberPlayers  bestMinNumberPlayers  bestMaxNumberPlayers  avgRating  numRatings  numComments  numFans
1     Die Macher  1/die-macher    1986  4.32    14   240          240          3                 5                 5                    5                    7.61       5418         2030         237

---

### 2. raw_game_metadata_2.tsv

Captures additional metadata such as game type, categories, mechanics, and families.

**Columns:**
- id: Game identifier
- property: Metadata type (type, category, mechanism, family)
- value: Value for the property

**Example:**
id  property   value
1   type       Strategy Games
1   category   Negotiation
1   category   Political
1   family     Country: Germany
1   mechanism  Alliances
1   mechanism  Dice Rolling

---

### 3. raw_game_descriptions.tsv

Provides textual descriptions of each game.

**Columns:**
- id: Game identifier
- description: Game description text

**Example:**
id  description
1   Die Macher is a game about seven sequential political races in different regions of Germany...

---

### 4. raw_game_rankpositions.tsv

Stores the ranking positions of games within different ranking categories.

**Columns:**
- id: Game identifier
- rank: Ranking category (e.g., Overall, Strategy)
- position: Rank position within the category

**Example:**
id  rank     position
1   Overall  328
1   Strategy 189

---

### 5. ctx_game_context_annotations.tsv

Includes sentence-level contextual annotations extracted from user reviews, along with polarity and negation indicators.

**Columns:**
- userId: ID of the user who wrote the review
- gameId: Game ID
- sentenceId: Position of the sentence in the review
- sentenceText: Original sentence
- contextTerm: Phrase extracted as a contextual clue
- contextTermNegated: Whether the term is negated (true/false)
- contextId: Context identifier
- contextDimension: Broad context dimension (e.g., gaming_mood)
- contextFactor: Fine-grained context category (e.g., expert, easy-going)
- contextPolarity: Sentiment polarity (positive, negative, neutral, mixed, non-contextual)

**Example:**
userId  gameId  sentenceId  sentenceText                                                         contextTerm               contextTermNegated  contextId  contextDimension  contextFactor  contextPolarity
36439   68448   2           One 6 player game and went straight into the 2P "Expert" game.       expert game              false                8          gaming_mood       expert         positive
7030    6249    1           Plays well with 4 players, not a difficult game to play.             not a difficult game     true                 7          gaming_mood       easy-going     positive

---

### 6. ctx_game_contexts_metadata.tsv

Provides metadata for each context dimension and factor associated with a game.

**Columns:**
- gameId: Game ID
- contextId: Context identifier
- contextName: Concatenated name of the context dimension and factor
- contextWeight: Weight or relevance of the context (usually binary or normalized count)

**Example:**
gameId  contextId  contextName                  contextWeight
1       11         gaming_mood:competitive      1

---

### 7. ctx_game_contexts_reviews.tsv

Captures the frequency or importance of context mentions in full reviews (as opposed to sentence-level annotations).

**Columns:**
- gameId: Game ID
- contextId: Context identifier
- contextName: Full name of the context
- contextWeight: Context relevance, typically frequency-based

**Example:**
gameId  contextId  contextName              contextWeight
1       5          playing_time:very_long   53

