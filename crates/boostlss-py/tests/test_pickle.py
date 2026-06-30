def test_pickle_logistic():
    from boostlss_py import LogisticLss, ZINBLss, LaplaceLss
    import pickle

    fam = LogisticLss()
    pickle.loads(pickle.dumps(fam))

    fam2 = ZINBLss()
    pickle.loads(pickle.dumps(fam2))

    fam3 = LaplaceLss()
    pickle.loads(pickle.dumps(fam3))
    print("Pickle success")


if __name__ == "__main__":
    test_pickle_logistic()
