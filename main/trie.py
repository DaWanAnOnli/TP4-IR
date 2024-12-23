# INFORMASI PENTING:
# Silakan merujuk pada slide di SCeLE tentang Query Auto-Completion (13-14)
# untuk referensi implementasi struktur data trie.

class TrieNode:
    """
    Abstraksi node dalam suatu struktur data trie.
    """
    def __init__(self, char):
        self.char = char
        self.freq = 0
        self.children = {}

    def __str__(self):
        return self.char

class Trie:
    """
    Abstraksi struktur data trie.
    """
    def __init__(self):
        self.root = TrieNode("")

    def insert(self, word, freq):
        node = self.root
        for char in word:
            if char in node.children:
                node = node.children[char]
            else:
                new_node = TrieNode(char)
                node.children[char] = new_node
                node = new_node
        node.freq += freq

    def __get_last_node(self, query):
        """
        Method ini mengambil node terakhir yang berasosiasi dengan suatu kata.
        Misalnya untuk query "halo", maka node terakhir adalah node "o"
        Jika no match, cukup return None saja.

        Parameters
        ----------
        query: str
            query yang ingin dilengkapi
        
        Returns
        -------
        TrieNode
            node terakhir dari suatu query, atau None
            jika tidak match
        """
        node = self.root
        print("node children: ", node.children)
        for char in query:
            if char in node.children:
                node = node.children[char]
            else:
                return None
        return node

    def __get_all_next_subwords(self, query):
        """
        Method ini melakukan traversal secara DFS untuk mendapatkan semua
        subwords yang mengikuti suatu query yang diberikan beserta dengan 
        frekuensi kemunculannya dalam struktur data dictionary. Silakan membuat
        fungsi helper jika memang dibutuhkan.

        Jika tidak ada match, return dictionary kosong saja.

        Parameters
        ----------
        query: str
            query yang ingin dilengkapi
        
        Returns
        -------
        dict(str, int)
            dictionary dengan key berupa kandidat subwords dan value berupa
            frekuensi kemunculan subwords tersebut
        """
        node = self.__get_last_node(query)
        if not node:
            return {}

        words = {}

        def dfs(current_node, current_prefix):
            if current_node.freq > 0:
                words[current_prefix] = current_node.freq
            for char, child_node in current_node.children.items():
                dfs(child_node, current_prefix + char)

        dfs(node, "")
        return words

    def get_recommendations(self, query, k=5):
        """
        Method ini mengembalikan top-k rekomendasi subwords untuk melanjutkan
        query yang diberikan. Urutkan berdasarkan value (frekuensi) kemunculan
        subwords secara descending.

        Parameters
        ----------
        query: str
            query yang ingin dilengkapi
        k: int
            top-k subwords yang paling banyak frekuensinya
        
        Returns
        -------
        List[str]
            top-k subwords yang paling "matched"
        """
        subwords = self.__get_all_next_subwords(query)
        if not subwords:
            return []

        # Urutkan subwords berdasarkan frekuensi secara descending
        sorted_subwords = sorted(subwords.items(), key=lambda item: item[1], reverse=True)

        # Ambil top-k hasil yang diinginkan, tanpa menambahkan prefiks query
        return [subword for subword, _ in sorted_subwords[:k]]

if __name__ == '__main__':
    # contoh dari slide
    trie = Trie()
    trie.insert("nba", 6)
    trie.insert("news", 6)
    trie.insert("nab", 8)
    trie.insert("ngv", 9)
    trie.insert("netflix", 7)
    trie.insert("netbank", 8)
    trie.insert("network", 1)
    trie.insert("netball", 3)
    trie.insert("netbeans", 4)

    assert trie.get_recommendations('n') == ['gv', 'etbank', 'ab', 'etflix', 'ba'], "output salah"
    assert trie.get_recommendations('') == ['ngv', 'netbank', 'nab', 'netflix', 'nba'], "output salah"
    assert trie.get_recommendations('a') == [], "output salah"
    assert trie.get_recommendations('na') == ['b'], "output salah"