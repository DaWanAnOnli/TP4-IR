class IdMap:
    """
    Ingat kembali di kuliah, bahwa secara praktis, sebuah dokumen dan
    sebuah term akan direpresentasikan sebagai sebuah integer. Oleh
    karena itu, kita perlu maintain mapping antara string term (atau
    dokumen) ke integer yang bersesuaian, dan sebaliknya. Kelas IdMap ini
    akan melakukan hal tersebut.
    """

    def __init__(self):
        """
        Mapping dari string (term atau nama dokumen) ke id disimpan dalam
        python's dictionary; cukup efisien. Mapping sebaliknya disimpan dalam
        python's list.

        contoh:
            str_to_id["halo"] ---> 8
            str_to_id["/collection/dir0/gamma.txt"] ---> 54

            id_to_str[8] ---> "halo"
            id_to_str[54] ---> "/collection/dir0/gamma.txt"
        """
        self.str_to_id = {}
        self.id_to_str = []

    def __len__(self):
        """Mengembalikan banyaknya term (atau dokumen) yang disimpan di IdMap."""
        return len(self.id_to_str)

    def __get_id(self, s):
        """
        Mengembalikan integer id i yang berkorespondensi dengan sebuah string s.
        Jika s tidak ada pada IdMap, lalu assign sebuah integer id baru dan kembalikan
        integer id baru tersebut.
        """
        if s not in self.str_to_id:
            self.id_to_str.append(s)
            self.str_to_id[s] = len(self.id_to_str)-1 
        return self.str_to_id[s]

    def __get_str(self, i):
        """Mengembalikan string yang terasosiasi dengan index i."""
        # print(self.id_to_str)
        # print(self.str_to_id)
        try:
            return self.id_to_str[i]
        except:
            raise KeyError("Id not found")

    def __getitem__(self, key):
        """
        __getitem__(...) adalah special method di Python, yang mengizinkan sebuah
        collection class (seperti IdMap ini) mempunyai mekanisme akses atau
        modifikasi elemen dengan syntax [..] seperti pada list dan dictionary di Python.

        Silakan search informasi ini di Web search engine favorit Anda. Saya mendapatkan
        link berikut:

        https://stackoverflow.com/questions/43627405/understanding-getitem-method

        Jika key adalah integer, gunakan __get_str;
        jika key adalah string, gunakan __get_id
        """
        if type(key) == str:
            return self.__get_id(key)
        elif type(key) == int:
            return self.__get_str(key)

        raise ValueError("Invalid argument for get item")


def merge_and_sort_posts_and_tfs(posts_tfs1, posts_tfs2):
    """
    Menggabungkan dua lists of tuples (doc_id, tf) yang sudah terurut,
    dan mengembalikan hasil penggabungan dengan TF yang terakumulasi
    untuk doc_id yang sama, dalam waktu O(n + m).
    """

    merged_list = []
    i, j = 0, 0

    while i < len(posts_tfs1) and j < len(posts_tfs2):
        doc_id1, tf1 = posts_tfs1[i]
        doc_id2, tf2 = posts_tfs2[j]

        if doc_id1 == doc_id2:
            merged_list.append((doc_id1, tf1 + tf2))
            i += 1
            j += 1
        elif doc_id1 < doc_id2:
            merged_list.append((doc_id1, tf1))
            i += 1
        else:
            merged_list.append((doc_id2, tf2))
            j += 1

    while i < len(posts_tfs1):
        merged_list.append(posts_tfs1[i])
        i += 1

    while j < len(posts_tfs2):
        merged_list.append(posts_tfs2[j])
        j += 1

    return merged_list



if __name__ == '__main__':

    doc = ["halo", "semua", "selamat", "pagi", "semua"]
    term_id_map = IdMap()

    assert [term_id_map[term]
            for term in doc] == [0, 1, 2, 3, 1], "term_id salah"
    assert term_id_map[1] == "semua", "term_id salah"
    assert term_id_map[0] == "halo", "term_id salah"
    assert term_id_map["selamat"] == 2, "term_id salah"
    assert term_id_map["pagi"] == 3, "term_id salah"

    # docs = ["/collection/0/data0.txt",
    #         "/collection/0/data10.txt",
    #         "/collection/1/data53.txt"]
    # doc_id_map = IdMap()
    # assert [doc_id_map[docname]
    #         for docname in docs] == [0, 1, 2], "docs_id salah"

    # assert merge_and_sort_posts_and_tfs([(1, 34), (3, 2), (4, 23)],
                                        # [(1, 11), (2, 4), (4, 3), (6, 13)]) == [(1, 45), (2, 4), (3, 2), (4, 26), (6, 13)], "merge_and_sort_posts_and_tfs salah"
