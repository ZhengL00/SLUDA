class Evaluator(object):
    def __init__(self):
        pass
    def compute_iou(self, predictedBoxList, gtBoxList):
        assert len(predictedBoxList) == len(
            gtBoxList
        ), "The list of predicted bounding boxes ({}) should be the same size as the list of ground truth bounding boxes ({}).".format(
            len(predictedBoxList), len(gtBoxList)
        )

        iouList = []
        for box1, box2 in zip(gtBoxList, predictedBoxList):
            iou = self._iou(box1, box2)
            iouList.append(iou)

        return iouList

    def accuracy(self, iouList, iouThreshold=0.5):

        matches = len([1 for iou in iouList if iou >= iouThreshold])
        accuracy = matches * 1.0 / len(iouList)
        return accuracy

    def evaluate(self, predictedBoxList, gtBoxList, iouThreshold=0.5):


        iouList = self.compute_iou(predictedBoxList, gtBoxList)
        accuracy = self.accuracy(iouList, iouThreshold)
        return (accuracy, iouList)

    def evaluate_perclass(
        self, predictedBoxList, gtBoxList, boxCategoriesList, iouThreshold=0.5
    ):

        categorySet = set()
        for categoryList in boxCategoriesList:
            categorySet.update(categoryList)

        iouList = self.compute_iou(predictedBoxList, gtBoxList)
        accuracy = self.accuracy(iouList, iouThreshold)
        perClassAccDict = {}
        for category in categorySet:
            subPredictedBoxList = []
            subGtBoxList = []
            for pred, gt, categoryList in zip(
                predictedBoxList, gtBoxList, boxCategoriesList
            ):
                if category in categoryList:
                    subPredictedBoxList.append(pred)
                    subGtBoxList.append(gt)
            subIouList = self.compute_iou(subPredictedBoxList, subGtBoxList)
            perClassAccDict[category] = self.accuracy(subIouList, iouThreshold)

        return (accuracy, perClassAccDict, iouList)

    def evaluate_upperbound_perclass(
        self, predictedBoxList, gtBoxList, boxCategoriesList, iouThreshold=0.5
    ):

        categorySet = set()
        for categoryList in boxCategoriesList:
            categorySet.update(categoryList)

        iouList = []
        argmaxList = []
        for i, gtBox in enumerate(gtBoxList):
            nCandidates = len(predictedBoxList[i])
            replicatedGtBoxList = []
            for j in range(nCandidates):
                replicatedGtBoxList.append(gtBox)
            instanceIouList = self.compute_iou(predictedBoxList[i], replicatedGtBoxList)
            maxIou = max(instanceIouList)
            iouList.append(maxIou)
            argmaxList.append(instanceIouList.index(maxIou))
        accuracy = self.accuracy(iouList, iouThreshold)
        perClassAccDict = {}
        for category in categorySet:
            subPredictedBoxList = []
            subGtBoxList = []
            for pred, gt, categoryList in zip(
                predictedBoxList, gtBoxList, boxCategoriesList
            ):
                if category in categoryList:
                    subPredictedBoxList.append(pred)
                    subGtBoxList.append(gt)
            subIouList = []
            for i, subGtBox in enumerate(subGtBoxList):
                nCandidates = len(subPredictedBoxList[i])
                replicatedGtBoxList = []
                for j in range(nCandidates):
                    replicatedGtBoxList.append(subGtBox)
                instanceIouList = self.compute_iou(
                    subPredictedBoxList[i], replicatedGtBoxList
                )
                maxIou = max(instanceIouList)
                subIouList.append(maxIou)

            perClassAccDict[category] = self.accuracy(subIouList, iouThreshold)

        return (accuracy, perClassAccDict, iouList, argmaxList)

    def _iou(self, box1, box2):
        (box1_left_x, box1_top_y, box1_right_x, box1_bottom_y) = box1
        box1_w = box1_right_x - box1_left_x + 1
        box1_h = box1_bottom_y - box1_top_y + 1

        (box2_left_x, box2_top_y, box2_right_x, box2_bottom_y) = box2
        box2_w = box2_right_x - box2_left_x + 1
        box2_h = box2_bottom_y - box2_top_y + 1
        intersect_left_x = max(box1_left_x, box2_left_x)
        intersect_top_y = max(box1_top_y, box2_top_y)
        intersect_right_x = min(box1_right_x, box2_right_x)
        intersect_bottom_y = min(box1_bottom_y, box2_bottom_y)

        overlap_x = max(0, intersect_right_x - intersect_left_x + 1)
        overlap_y = max(0, intersect_bottom_y - intersect_top_y + 1)
        intersect = overlap_x * overlap_y

        union = (box1_w * box1_h) + (box2_w * box2_h) - intersect

        return intersect * 1.0 / union
