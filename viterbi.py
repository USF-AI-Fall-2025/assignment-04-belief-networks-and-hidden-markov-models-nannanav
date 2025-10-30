from collections import defaultdict
import math

class Viterbi:
    def readfile(self, filename):
        with open(filename, 'r') as file:
            lines = file.readlines()
        self.lines = lines

    def calculate_emission_probabilities(self):
        emission_counts = defaultdict(lambda: defaultdict(int))
        for line in self.lines:
            words = line.split()
            correct_word = words[0][:-1]
            typed_words = words[1:]
            for typed_word in typed_words:
                for i, c in enumerate(correct_word):
                    if i < len(typed_word):
                        tc = typed_word[i]
                    else:
                        tc = ''
                    emission_counts[c][tc] += 1
                    emission_counts[c]['count'] += 1
        self.emission_counts = emission_counts

    def calculate_transition_probabilities(self):
        transition_counts = defaultdict(lambda: defaultdict(int))
        for line in self.lines:
            words = line.split()
            word = words[0][:-1]
            for i, c in enumerate(word):
                if i == 0:
                    transition_counts["<start>"][c] += 1
                    transition_counts["<start>"]["count"] += 1
                if i+1 == len(word):
                    transition_counts[c]["<end>"] += 1
                    transition_counts[c]["count"] += 1
                else:
                    transition_counts[c][word[i+1]] += 1
                    transition_counts[c]["count"] += 1
        self.transition_counts = transition_counts

    def viterbi_decode(self, typed_word):
        """
        States = correct letters, Observations = typed letters
        Uses log probabilities to avoid numerical underflow.
        """
        # Get all possible states (correct letters that appear in training data)
        states = list(self.emission_counts.keys())
        
        # if not typed_word or not states:
        #     return typed_word
        
        n = len(typed_word)
        
        # Initialize M[t,s] (now in log space) and Backpointers[t,s]
        M = defaultdict(lambda: defaultdict(lambda: float('-inf')))
        backpointers = defaultdict(lambda: defaultdict(str))
        
        # Set up initial probabilities from start state to first observation
        # M[1,s] = log(T[start, s]) + log(E[s, O[1]])
        for s in states:
            trans_count = self.transition_counts["<start>"].get(s, 0)
            trans_total = max(self.transition_counts["<start>"].get("count", 1), 1)
            trans_prob = trans_count / trans_total if trans_count > 0 else 1e-10
            
            emit_count = self.emission_counts[s].get(typed_word[0], 0)
            emit_total = max(self.emission_counts[s].get('count', 1), 1)
            emit_prob = emit_count / emit_total if emit_count > 0 else 1e-10
            
            # Store log probabilities
            M[1][s] = math.log(trans_prob) + math.log(emit_prob)
        
        # For each time step (observation)
        for t in range(2, n + 1):
            observation_idx = t - 1  # Convert to 0-indexed
            
            # For each possible hidden state
            for s in states:
                max_log_prob = float('-inf')
                best_prev_state = None
                
                # Find max over all previous states
                # log_val = log(M[state2, t-1]) + log(T[state2, s]) + log(E[s, O[t]])
                # In log space: log_val = M[state2, t-1] + log(T[state2, s]) + log(E[s, O[t]])
                for state2 in states:
                    prev_log_prob = M[t-1][state2]
                    
                    if prev_log_prob == float('-inf'):
                        continue
                    
                    trans_count = self.transition_counts[state2].get(s, 0)
                    trans_total = max(self.transition_counts[state2].get("count", 1), 1)
                    trans_prob = trans_count / trans_total if trans_count > 0 else 1e-10
                    
                    emit_count = self.emission_counts[s].get(typed_word[observation_idx], 0)
                    emit_total = max(self.emission_counts[s].get('count', 1), 1)
                    emit_prob = emit_count / emit_total if emit_count > 0 else 1e-10
                    
                    # Addition in log space (equivalent to multiplication in normal space)
                    log_val = prev_log_prob + math.log(trans_prob) + math.log(emit_prob)
                    
                    if log_val > max_log_prob:
                        max_log_prob = log_val
                        best_prev_state = state2
                
                M[t][s] = max_log_prob
                backpointers[t][s] = best_prev_state
        
        # Find the best final state
        # Best = max(M[T,:])
        max_log_prob = float('-inf')
        best_state = None
        for s in states:
            if M[n][s] > max_log_prob:
                max_log_prob = M[n][s]
                best_state = s
        
        if best_state is None:
            return typed_word
        
        # Work backwards through time to reconstruct the path
        path = []
        current = best_state
        for t in range(n, 0, -1):
            path.append(current)
            if t > 1:
                current = backpointers[t][current]
        
        # Reverse to get correct order
        path.reverse()
        return ''.join(path)

    def test(self):
        while True:
            try:
                line = input()
                words = line.split()
                decoded_words = []
                for word in words:
                    # Use the Viterbi algorithm to decode each typed word
                    decoded_word = self.viterbi_decode(word)
                    decoded_words.append(decoded_word)
                print(' '.join(decoded_words))
            except EOFError:
                break
            except KeyboardInterrupt:
                break



if __name__ == "__main__":
    filename = "aspell.txt"
    viterbi = Viterbi()
    viterbi.readfile(filename)
    viterbi.calculate_emission_probabilities()
    viterbi.calculate_transition_probabilities()
    viterbi.test()