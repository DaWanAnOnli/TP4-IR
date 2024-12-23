import array

# PERHATIAN: TF List di-encode juga dengan metode yang sama dengan Postings List

class StandardPostings:
    """ 
    Class dengan static methods, untuk mengubah representasi postings list
    yang awalnya adalah List of integer, berubah menjadi sequence of bytes.
    Kita menggunakan Library array di Python.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!

    Silakan pelajari:
        https://docs.python.org/3/library/array.html
    """

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
        """
        # Untuk yang standard, gunakan L untuk unsigned long, karena docID
        # tidak akan negatif. Dan kita asumsikan docID yang paling besar
        # cukup ditampung di representasi 4 byte unsigned.
        return array.array('L', postings_list).tobytes()

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari sebuah stream of bytes

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        decoded_postings_list = array.array('L')
        decoded_postings_list.frombytes(encoded_postings_list)
        return decoded_postings_list.tolist()

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies menjadi stream of bytes

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            bytearray yang merepresentasikan nilai raw TF kemunculan term di setiap
            dokumen pada list of postings
        """
        return StandardPostings.encode(tf_list)

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decodes list of term frequencies dari sebuah stream of bytes

        Parameters
        ----------
        encoded_tf_list: bytes
            bytearray merepresentasikan encoded term frequencies list sebagai keluaran
            dari static method encode_tf di atas.

        Returns
        -------
        List[int]
            List of term frequencies yang merupakan hasil decoding dari encoded_tf_list
        """
        return StandardPostings.decode(encoded_tf_list)


class VBEPostings:
    """ 
    Berbeda dengan StandardPostings, dimana untuk suatu postings list,
    yang disimpan di disk adalah sequence of integers asli dari postings
    list tersebut apa adanya.

    Pada VBEPostings, kali ini, yang disimpan adalah gap-nya, kecuali
    posting yang pertama. Barulah setelah itu di-encode dengan Variable-Byte
    Enconding algorithm ke bytestream.

    Contoh:
    postings list [34, 67, 89, 454] akan diubah dulu menjadi gap-based,
    yaitu [34, 33, 22, 365]. Barulah setelah itu di-encode dengan algoritma
    compression Variable-Byte Encoding, dan kemudian diubah ke bytesream.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!

    """

    @staticmethod
    def vb_encode_number(number):
        """
        Encodes a number using Variable-Byte Encoding
        Lihat buku teks kita!
        """
        bytes = []
        while True:
            bytes.insert(0, number % 128)
            if number < 128:
                break
            number = number // 128
        bytes[-1] += 128

        return bytes

    @staticmethod
    def vb_encode(list_of_numbers):
        """ 
        Melakukan encoding (tentunya dengan compression) terhadap
        list of numbers, dengan Variable-Byte Encoding
        """
        bytestream = []
        for number in list_of_numbers:
            bytes = VBEPostings.vb_encode_number(number)
            bytestream += bytes
        return bytestream

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes (dengan Variable-Byte
        Encoding). JANGAN LUPA diubah dulu ke gap-based list, sebelum
        di-encode dan diubah ke bytearray.

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
        """
        list_of_numbers = [postings_list[0]]
        for i in range(1, len(postings_list)):
            list_of_numbers.append(postings_list[i] - postings_list[i-1])
        result = VBEPostings.vb_encode(list_of_numbers)
        return array.array('B', result).tobytes()

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies menjadi stream of bytes

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            bytearray yang merepresentasikan nilai raw TF kemunculan term di setiap
            dokumen pada list of postings
        """

        result = VBEPostings.vb_encode(tf_list)
        return array.array('B', result).tobytes()

    @staticmethod
    def vb_decode(encoded_bytestream):
        """
        Decoding sebuah bytestream yang sebelumnya di-encode dengan
        variable-byte encoding.
        """
        numbers = []
        n = 0
        for byte in encoded_bytestream:
            if (byte < 128):
                n = 128 * n + byte
            else:
                n = 128 * n + byte - 128
                numbers.append(n)
                n = 0
        return numbers

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari sebuah stream of bytes. JANGAN LUPA
        bytestream yang di-decode dari encoded_postings_list masih berupa
        gap-based list.

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        encoded_bytestream = array.array('B')
        encoded_bytestream.frombytes(encoded_postings_list)

        decoded_gap_list = VBEPostings.vb_decode(encoded_bytestream)
        if len(decoded_gap_list) == 0:
            return []
    
        decoded_postings_list = [decoded_gap_list[0]]  # Elemen pertama tetap sama
        for i in range(1,len(decoded_gap_list)):
            decoded_postings_list.append(decoded_postings_list[-1] + decoded_gap_list[i])  # Tambahkan gap ke elemen sebelumnya

        return decoded_postings_list

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decodes list of term frequencies dari sebuah stream of bytes

        Parameters
        ----------
        encoded_tf_list: bytes
            bytearray merepresentasikan encoded term frequencies list sebagai keluaran
            dari static method encode_tf di atas.

        Returns
        -------
        List[int]
            List of term frequencies yang merupakan hasil decoding dari encoded_tf_list
        """
        encoded_bytestream = array.array('B')
        encoded_bytestream.frombytes(encoded_tf_list)

        decoded_tf_list = VBEPostings.vb_decode(encoded_bytestream)
        if len(decoded_tf_list) == 0:
            return []

        return decoded_tf_list


if __name__ == '__main__':

    postings_list = [34, 67, 89, 454, 2345738]
    tf_list = [12, 10, 3, 4, 1]
    for Postings in [StandardPostings, VBEPostings]:
        print(Postings.__name__)
        encoded_postings_list = Postings.encode(postings_list)
        encoded_tf_list = Postings.encode_tf(tf_list)
        print("byte hasil encode postings: ", encoded_postings_list)
        print("ukuran encoded postings   : ",
              len(encoded_postings_list), "bytes")
        print("byte hasil encode TF list : ", encoded_tf_list)
        print("ukuran encoded postings   : ", len(encoded_tf_list), "bytes")

        decoded_posting_list = Postings.decode(encoded_postings_list)
        decoded_tf_list = Postings.decode_tf(encoded_tf_list)
        print("hasil decoding (postings): ", decoded_posting_list)
        print("hasil decoding (TF list) : ", decoded_tf_list)
        assert decoded_posting_list == postings_list, "hasil decoding tidak sama dengan postings original"
        print(decoded_tf_list, tf_list)
        assert decoded_tf_list == tf_list, "hasil decoding tidak sama dengan postings original"
        print()
