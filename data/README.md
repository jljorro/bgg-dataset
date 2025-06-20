# BGG Datasets

This folder contains four datasets with information about user interactions with board games from the BoardGameGeek platform. Each dataset includes the following fields: a user identifier (`user_id`), a game identifier (`game_id`), a rating (`rating`), and a timestamp (`timestamp`) indicating when the interaction took place.

In addition, all three datasets include 21 contextual attributes that describe aspects of the interaction or the items involved. These contextual attributes are grouped into three contextual dimensions as follows:

- **Playing time**: `playing_time_very_short`, `playing_time_short`, `playing_time_moderate`, `playing_time_long` and `playing_time_very_long`.
- **Gaming mood**: `gaming_mood_party`, `gaming_mood_easy-going`, `gaming_mood_expert`, `gaming_mood_intense`, `gaming_mood_cooperative`, `gaming_mood_competitive`, `gaming_mood_thematic`, `gaming_mood_story-based`.
- **Social companion**: `social_companion_1-player`, `social_companion_2-players`, `social_companion_large-group`, `social_companion_toddlers`, `social_companion_preschoolers`, `social_companion_children`, `social_companion_family`, `social_companion_friends`.

These context features are represented in different formats across the datasets to support various experimental settings and model configurations

## Raw Ratings

- **Folder**: `bgg25_raw_ratings`. 

This dataset does not include any contextual information. It is used to train and evaluate general (non-context-aware) recommender models.

## Continuous Context Values Metadata

- **Folder**: `bgg25_continuous_metadata`. 

This dataset includes contextual information extracted from the games' metadata. Each context attribute is represented as a `float` value (0.0 or 1.0). Thus, attributes are not converted to internal embeddings in RecBole.

## Discrete Context Values Metadata

- **Folder**: `bgg25_discrete_metadata`. 

This dataset also includes contextual information from the games' metadata, but each context attribute is represented as a discrete `token` (e.g., 0 or 1). This allows the use of embeddings in RecBole models by treating attributes as categorical variables.

## Continuous Context Values from Reviews

- **Folder**: `bgg25_continuous_reviews`. 

This dataset includes contextual information extracted from user reviews on the BGG platform. Each context attribute is represented as a `float` value ranging between 0.0 and 1.0.
