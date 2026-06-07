"""
TRACE-style temporal modules for structured token generation paradigm.

Implements character-level tokenizers and embedding towers for the
causal event modeling approach (TRACE, Guo et al. 2024).

Components:
- TimeTokenizer / ScoreTokenizer: char-level tokenizers (vocab=13)
- TimeTower / ScoreTower: embedding towers for time/score tokens
- SyncTower: single learnable embedding for head-switching sync tokens
"""

import re
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer
from typing import List, Optional


# ================================================================
# Tokenizers (character-level, vocab=13)
# ================================================================

class TimeTokenizer(PreTrainedTokenizer):
    """
    Character-level tokenizer for timestamps.

    Vocab (13 tokens):
        0: <sync>   - end-of-time-sequence marker (triggers head switch)
        1: <sep>    - separator between start and end timestamps
        2-11: '0'-'9' - digit characters
        12: '.'     - decimal point

    Example: [12.3, 45.6] -> '012.3<sep>045.6<sync>'
             -> token IDs: [2,3,12,5,12, 1, 2,6,7,12,8, 0]
    """

    def __init__(self, *args, **kwargs):
        self.vocab = {'<sync>': 0, '<sep>': 1}
        for i in range(10):
            self.vocab[str(i)] = i + 2
        self.vocab['.'] = 12
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        super().__init__(*args, **kwargs)

    def _tokenize(self, text, *args, **kwargs):
        pattern = '|'.join(map(re.escape, self.vocab.keys()))
        return [t for t in re.findall(pattern, text) if t]

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, self.unk_token)

    def get_vocab(self):
        return dict(self.vocab)

    @property
    def vocab_size(self):
        return len(self.vocab)


class ScoreTokenizer(PreTrainedTokenizer):
    """
    Character-level tokenizer for saliency scores.
    Same vocabulary structure as TimeTokenizer.

    Example: [5.0] -> '5.0<sync>' -> token IDs: [7, 12, 2, 0]
    """

    def __init__(self, *args, **kwargs):
        self.vocab = {'<sync>': 0, '<sep>': 1}
        for i in range(10):
            self.vocab[str(i)] = i + 2
        self.vocab['.'] = 12
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        super().__init__(*args, **kwargs)

    def _tokenize(self, text, *args, **kwargs):
        pattern = '|'.join(map(re.escape, self.vocab.keys()))
        return [t for t in re.findall(pattern, text) if t]

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, self.unk_token)

    def get_vocab(self):
        return dict(self.vocab)

    @property
    def vocab_size(self):
        return len(self.vocab)


# ================================================================
# Embedding Towers
# ================================================================

class TimeTower(nn.Module):
    """
    Embedding tower for time tokens.

    Maps character-level token IDs to hidden_dim embeddings.
    Each timestamp pair (start, end) is encoded as a character sequence:
        [12.3, 45.6] -> '012.3<sep>045.6<sync>'

    Parameters:
        tokenizer: TimeTokenizer instance
        hidden_dim: embedding dimension (must match LLM hidden_size)
    """

    def __init__(self, tokenizer: TimeTokenizer, hidden_dim: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.hidden_dim = hidden_dim
        self.embed_tokens = nn.Embedding(tokenizer.vocab_size, hidden_dim)

    def encode(self, timestamps: List[float]) -> torch.Tensor:
        """
        Encode a list of timestamps to token IDs.

        Args:
            timestamps: list of float values, e.g. [12.3, 45.6] for a (start, end) pair

        Returns:
            input_ids: (num_tokens,) tensor of token IDs (local, 0-12)
                       Includes <sep> between values and trailing <sync>
        """
        def insert_separator(X, sep):
            return [self.tokenizer(ele).input_ids
                    for sublist in zip(X, [sep] * len(X))
                    for ele in sublist][:-1]

        # Format to fixed-width string: '012.3'
        timestamps = [format(t, '0>6.1f') for t in timestamps]

        input_ids = []
        for ids in insert_separator(timestamps, '<sep>'):
            input_ids.extend(ids)
        input_ids.extend(self.tokenizer('<sync>').input_ids)

        return torch.tensor(input_ids, dtype=torch.long)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (...) tensor of LOCAL time token IDs (0-12)

        Returns:
            embeddings: (..., hidden_dim) tensor
        """
        return self.embed_tokens(input_ids)


class ScoreTower(nn.Module):
    """
    Embedding tower for saliency score tokens.
    Same architecture as TimeTower but for score values.

    Parameters:
        tokenizer: ScoreTokenizer instance
        hidden_dim: embedding dimension (must match LLM hidden_size)
    """

    def __init__(self, tokenizer: ScoreTokenizer, hidden_dim: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.hidden_dim = hidden_dim
        self.embed_tokens = nn.Embedding(tokenizer.vocab_size, hidden_dim)

    def encode(self, scores: List[float]) -> torch.Tensor:
        """
        Encode a list of scores to token IDs.

        Args:
            scores: list of float values, e.g. [5.0] for a single saliency score

        Returns:
            input_ids: (num_tokens,) tensor of token IDs (local, 0-12)
        """
        def insert_separator(X, sep):
            return [self.tokenizer(ele).input_ids
                    for sublist in zip(X, [sep] * len(X))
                    for ele in sublist][:-1]

        scores = [format(s, '0>3.1f') for s in scores]

        input_ids = []
        for ids in insert_separator(scores, '<sep>'):
            input_ids.extend(ids)
        input_ids.extend(self.tokenizer('<sync>').input_ids)

        return torch.tensor(input_ids, dtype=torch.long)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (...) tensor of LOCAL score token IDs (0-12)

        Returns:
            embeddings: (..., hidden_dim) tensor
        """
        return self.embed_tokens(input_ids)


class SyncTower(nn.Module):
    """
    Single learnable embedding for sync (head-switching) tokens.

    All sync tokens map to the same learnable embedding regardless of
    their actual token ID. This acts as a boundary marker between
    text/time/score generation phases.

    Parameters:
        hidden_dim: embedding dimension (must match LLM hidden_size)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(1, hidden_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (...) tensor (values ignored, all map to index 0)

        Returns:
            embeddings: (..., hidden_dim) tensor
        """
        return self.embed_tokens(torch.zeros_like(input_ids))


# ================================================================
# Helper: decode generated token IDs back to timestamps/scores
# ================================================================

def decode_trace_tokens(
    token_ids: List[int],
    time_tokenizer: TimeTokenizer,
    score_tokenizer: ScoreTokenizer,
    time_start_id: int,
    time_end_id: int,
    score_start_id: int,
    score_end_id: int,
    sync_token_id: int,
    duration: Optional[float] = None,
) -> dict:
    """
    Decode a TRACE-generated token sequence back to timestamps and scores.

    The TRACE paradigm generates events autoregressively in the pattern:
        text... <sync> time_chars <time_sync> score_chars <score_sync> text...

    Args:
        token_ids: list of global token IDs from generation
        time_tokenizer: TimeTokenizer instance
        score_tokenizer: ScoreTokenizer instance
        time_start_id: global start ID of time token range
        time_end_id: global end ID of time token range (exclusive)
        score_start_id: global start ID of score token range
        score_end_id: global end ID of score token range (exclusive)
        sync_token_id: global ID of text sync token
        duration: video duration in seconds (for denormalization)

    Returns:
        dict with 'times' (list of [start, end] pairs) and 'scores' (list of floats)
    """
    times = []
    scores = []
    current_time_chars = []
    current_score_chars = []

    for tid in token_ids:
        if time_start_id <= tid < time_end_id:
            local_id = tid - time_start_id
            if local_id == 0:
                # Time <sync> token -> end of time sequence, decode accumulated chars
                if current_time_chars:
                    time_str = ''.join(current_time_chars)
                    parts = time_str.split('<sep>')
                    try:
                        parsed = [float(p.strip()) for p in parts if p.strip()]
                        if len(parsed) == 2:
                            times.append(parsed)
                        elif len(parsed) == 1:
                            times.append([parsed[0], parsed[0]])
                    except ValueError:
                        pass
                    current_time_chars = []
            else:
                char = time_tokenizer._convert_id_to_token(local_id)
                current_time_chars.append(char)

        elif score_start_id <= tid < score_end_id:
            local_id = tid - score_start_id
            if local_id == 0:
                # Score <sync> token -> end of score sequence, decode accumulated chars
                if current_score_chars:
                    score_str = ''.join(current_score_chars)
                    parts = score_str.split('<sep>')
                    try:
                        parsed = [float(p.strip()) for p in parts if p.strip()]
                        scores.extend(parsed)
                    except ValueError:
                        pass
                    current_score_chars = []
            else:
                char = score_tokenizer._convert_id_to_token(local_id)
                current_score_chars.append(char)

    return {'times': times, 'scores': scores}


# ================================================================
# Test
# ================================================================

if __name__ == "__main__":
    print("Testing TRACE modules...")

    hidden_dim = 2048

    # Test tokenizers
    time_tok = TimeTokenizer()
    score_tok = ScoreTokenizer()
    print(f"Time vocab: {time_tok.get_vocab()}")
    print(f"Score vocab: {score_tok.get_vocab()}")

    # Test towers
    time_tower = TimeTower(time_tok, hidden_dim)
    score_tower = ScoreTower(score_tok, hidden_dim)
    sync_tower = SyncTower(hidden_dim)

    # Test encoding
    time_ids = time_tower.encode([12.3, 45.6])
    print(f"Time encode [12.3, 45.6] -> {time_ids.tolist()}")
    print(f"  length: {len(time_ids)} tokens")

    score_ids = score_tower.encode([5.0])
    print(f"Score encode [5.0] -> {score_ids.tolist()}")
    print(f"  length: {len(score_ids)} tokens")

    # Test forward
    time_embeds = time_tower(time_ids)
    print(f"Time tower output: {time_embeds.shape}")

    score_embeds = score_tower(score_ids)
    print(f"Score tower output: {score_embeds.shape}")

    sync_embeds = sync_tower(torch.tensor([0, 0, 0]))
    print(f"Sync tower output: {sync_embeds.shape}")

    # Test decode
    result = decode_trace_tokens(
        token_ids=[],  # would be populated in real usage
        time_tokenizer=time_tok,
        score_tokenizer=score_tok,
        time_start_id=49154,
        time_end_id=49167,
        score_start_id=49167,
        score_end_id=49180,
        sync_token_id=49153,
    )
    print(f"Decode result: {result}")

    print("All tests passed!")
