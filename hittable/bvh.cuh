#pragma once

#include "hittable_list.cuh"
#include "sort.cuh"

#include <iostream>

class BvhNode :public Hittable {
  public:
    __host__ __device__ BvhNode() {}
    __host__ __device__ BvhNode(Hittable **objects, int n, float time0, float time1) : objects{objects}, nObjects{n}, time0{time0}, time1{time1} {}

    __host__ __device__ virtual bool hit(const Ray &r, float tMin, float tMax, HitRecord *rec, MaterialRecord *mat) const override;
    __host__ __device__ virtual float getNumCompare(int index) const override;
    __host__ __device__ virtual bool buildBoundingBox(BoundingRecord *box) override;

    __device__ static BvhNode* constructBvh(Hittable **objects, int n, float time0, float time1, curandState* randState);
    __device__ void sortDivide(curandState* randState);

    __host__ __device__
    Hittable** getList() {
      return this->objects;
    }

    __host__ __device__
    int getNumList() {
      return this->nObjects;
    }

    __host__ __device__
    Hittable* getLeft() {
      return this->leftObject;
    }

    __host__ __device__
    Hittable* getRight() {
      return this->rightObject;
    }

    private:
      Hittable **objects; int nObjects;
      Hittable *leftObject, *rightObject;
      AABB nodeBox;

      float time0, time1;

    __host__ __device__ static bool boxCompare(Hittable *prevObject, Hittable *nextObject, int axis);

    __host__ __device__ static bool boxCompareX(Hittable *prevObject, Hittable *nextObject);
    __host__ __device__ static bool boxCompareY(Hittable *prevObject, Hittable *nextObject);
    __host__ __device__ static bool boxCompareZ(Hittable *prevObject, Hittable *nextObject);
};

__host__ __device__ 
bool BvhNode::hit(const Ray &r, float tMin, float tMax, HitRecord *rec, MaterialRecord *mat) const {
  if (!this->nodeBox.hit(r, tMin, tMax)) {
    return false;
  }

  bool hitLeft = this->leftObject->hit(r, tMin, tMax, rec, mat);
  bool hitRight = this->rightObject->hit(r, tMin, hitLeft ? rec->t : tMax, rec, mat);

  return hitLeft || hitRight;
}

__host__ __device__ 
float BvhNode::getNumCompare(int index) const {
  return 0.0f;
}

__host__ __device__
bool BvhNode::buildBoundingBox(BoundingRecord *box) {
  BoundingRecord boxLeft, boxRight;

  this->leftObject->buildBoundingBox(&boxLeft);
  this->rightObject->buildBoundingBox(&boxRight);

  this->nodeBox = AABB::surrondingBox(boxLeft.boundingBox, boxRight.boundingBox);

  if (box != nullptr && box != NULL) {
    box->boundingBox = this->nodeBox;
  }

  return true;
}

__device__
void BvhNode::sortDivide(curandState* randState) {
  int axis = randInt(0, 2, randState);

  auto comparator = (axis == 0) ? boxCompareX
                  : (axis == 1) ? boxCompareY
                  : boxCompareZ;

  if (this->nObjects == 1) {
    this->leftObject = this->rightObject = this->objects[0];
  }

  else if (this->nObjects == 2) {
    if (comparator(this->objects[0], this->objects[1])) {
      this->leftObject = this->objects[0];
      this->rightObject = this->objects[1];
    }
    else {
      this->leftObject = this->objects[1];
      this->rightObject = this->objects[0];
    }
  }

  else {
    quickSortIterative<Hittable*>(this->objects, 0, this->nObjects - 1, comparator);
    auto mid = static_cast<int>(this->nObjects / 2);

    Hittable **leftObjects = (Hittable**) malloc(mid * sizeof(Hittable*));
    int nLeft = 0;

    for (int i = 0; i < mid; i++) {
      leftObjects[nLeft++] = this->objects[i];
    }

    Hittable **rightObjects = (Hittable**) malloc((this->nObjects - mid) * sizeof(Hittable*));
    int nRight = 0;

    for (int i = mid; i < this->nObjects; i++) {
      rightObjects[nRight++] = this->objects[i];
    }

    this->leftObject = new BvhNode(leftObjects, nLeft, time0, time1);
    this->rightObject = new BvhNode(rightObjects, nRight, time0, time1); 
  }
}

__host__ __device__
bool BvhNode::boxCompare(Hittable *prevObject, Hittable *nextObject, int axis) {
  return prevObject->getNumCompare(axis) < nextObject->getNumCompare(axis);
}

__host__ __device__
bool BvhNode::boxCompareX(Hittable *prevObject, Hittable *nextObject) {
  return BvhNode::boxCompare(prevObject, nextObject, 0);
}

__host__ __device__
bool BvhNode::boxCompareY(Hittable *prevObject, Hittable *nextObject) {
  return BvhNode::boxCompare(prevObject, nextObject, 1);
}

__host__ __device__
bool BvhNode::boxCompareZ(Hittable *prevObject, Hittable *nextObject) {
  return BvhNode::boxCompare(prevObject, nextObject, 2);
}

__host__ __device__
bool isNotLeaf(BvhNode **objects, int nObject) {
  for (int i = 0; i < nObject; i++) {
    if (objects[i]->getNumList() > 2) return true;
  }

  return false;
}

__device__
BvhNode* BvhNode::constructBvh(Hittable **objects, int n, float time0, float time1, curandState* randState) {
  BvhNode *root = new BvhNode(objects, n, time0, time1);
  root->sortDivide(randState);

  BvhNode **leafNodes = (BvhNode**) malloc(n * sizeof(BvhNode*));
  int nLeaf = 0;

  leafNodes[nLeaf++] = root;

  while (isNotLeaf(leafNodes, nLeaf)) {
    BvhNode **curNodes =  (BvhNode**) malloc(n * sizeof(BvhNode*));
    int nCurNodes = 0;

    for (int i = 0; i < nLeaf; i++) {
      curNodes[nCurNodes++] = leafNodes[i];
    }

    for (int i = 0; i < nCurNodes; i++) {
      if (curNodes[i]->getNumList() > 2) {
        BvhNode *leftNode = (BvhNode*) curNodes[i]->getLeft();
        BvhNode *rightNode = (BvhNode*) curNodes[i]->getRight();
        
        leafNodes[nLeaf++] = leftNode;
        leafNodes[nLeaf++] = rightNode;
      }

      for (int i = 1; i < nLeaf; i++) {
        leafNodes[i - 1] = leafNodes[i];
      }

      nLeaf--;
    }

    for (int i = 0; i < nLeaf; i++) {
      leafNodes[i]->sortDivide(randState);
    }
    
    free(curNodes);
  }

  root->buildBoundingBox(nullptr);
  return root;
}