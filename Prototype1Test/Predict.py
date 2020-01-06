def getSent(model, iters, minLength=1):
    sentences = {}
    for i in range(iters):
        modelGen = model.chain.gen()
        prevPrevWord = "___BEGIN__"
        prevWord = modelGen.__next__()
        madeSentence = prevWord + " "

        totalScore = 0
        numWords = 1
        for curWord in modelGen:
            madeSentence += curWord + " "
            numWords += 1
            totalScore += model.chain.model[(prevPrevWord, prevWord)][curWord]
            prevPrevWord = prevWord
            prevWord = curWord

        madeSentence = madeSentence.strip()
        if numWords == 0: continue

        # Commonly occurring words can be ignored
        if "am a bot" in madeSentence.lower(): continue
        if "questions or concerns" in madeSentence.lower(): continue
        if "contact the moderators" in madeSentence.lower(): continue
        if "this action was performed" in madeSentence.lower(): continue
        if numWords < minLength: continue
        if madeSentence in sentences: continue

        totalScore += model.chain.model[(prevPrevWord, prevWord)]["___END__"]

        sentences[madeSentence] = totalScore / float(numWords)

    # Get the sentences as (sentence, score) pairs
    sentences = sentences.items()
    # Sort them so the sentences with the highest score appear first

    sorted(sentences, key=lambda x: -x[1])

    return sentences
