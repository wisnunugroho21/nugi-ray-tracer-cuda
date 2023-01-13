#pragma once

#include "hittable_list.cuh"
#include "helper/sort.cuh"

__host__ __device__
bool boxCompare(Hittable *prevObject, Hittable *nextObject, int axis) {
  return prevObject->numCompare(axis) < nextObject->numCompare(axis);
}

__host__ __device__
bool boxCompareX(Hittable *prevObject, Hittable *nextObject) {
  return boxCompare(prevObject, nextObject, 0);
}

__host__ __device__
bool boxCompareY(Hittable *prevObject, Hittable *nextObject) {
  return boxCompare(prevObject, nextObject, 1);
}

__host__ __device__
bool boxCompareZ(Hittable *prevObject, Hittable *nextObject) {
  return boxCompare(prevObject, nextObject, 2);
}

class BvhNode :public Hittable {
  public:
    __host__ __device__ BvhNode() {}
    __host__ __device__ BvhNode(Hittable **objects, int n) : objects{objects}, nObjects{n} {}

    __device__ virtual bool hit(const Ray &r, float tMin, float tMax, HitRecord *rec, MaterialRecord *mat, curandState* randState) const override;
    __host__ virtual bool hit(const Ray &r, float tMin, float tMax, HitRecord *rec, MaterialRecord *mat) const override;

    __host__ __device__ virtual float numCompare(int index) const override;
    __host__ __device__ virtual bool boundingBox(BoundingRecord *box) override;
    __host__ virtual Hittable* copyToDevice() override;

    __device__ static BvhNode* build(Hittable **objects, int n, curandState* randState);
    __device__ void sortDivide(curandState* randState);

    __host__ static BvhNode* build(Hittable **objects, int n);
    __host__ void sortDivide();

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

    public:
      Hittable **objects; int nObjects;
      Hittable *leftObject, *rightObject;
      AABB nodeBox;
};

__device__ 
bool BvhNode::hit(const Ray &r, float tMin, float tMax, HitRecord *rec, MaterialRecord *mat, curandState* randState) const {
  if (!this->nodeBox.hit(r, tMin, tMax)) {
    return false;
  }

  bool hitLeft = this->leftObject->hit(r, tMin, tMax, rec, mat, randState);
  bool hitRight = this->rightObject->hit(r, tMin, hitLeft ? rec->t : tMax, rec, mat, randState);

  return hitLeft || hitRight;
}

__host__ 
bool BvhNode::hit(const Ray &r, float tMin, float tMax, HitRecord *rec, MaterialRecord *mat) const {
  if (!this->nodeBox.hit(r, tMin, tMax)) {
    return false;
  }

  bool hitLeft = this->leftObject->hit(r, tMin, tMax, rec, mat);
  bool hitRight = this->rightObject->hit(r, tMin, hitLeft ? rec->t : tMax, rec, mat);

  return hitLeft || hitRight;
}

__host__ __device__ 
float BvhNode::numCompare(int index) const {
  return 0.0f;
}

__host__ __device__
bool BvhNode::boundingBox(BoundingRecord *box) {
  BoundingRecord boxLeft, boxRight;

  this->leftObject->boundingBox(&boxLeft);
  this->rightObject->boundingBox(&boxRight);

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

    this->leftObject = new BvhNode(leftObjects, nLeft);
    this->rightObject = new BvhNode(rightObjects, nRight); 
  }
}

__host__
void BvhNode::sortDivide() {
  int axis = randInt(0, 2);

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

    this->leftObject = new BvhNode(leftObjects, nLeft);
    this->rightObject = new BvhNode(rightObjects, nRight); 
  }
}

__host__ __device__
bool isNotLeaf(BvhNode **objects, int nObject) {
  for (int i = 0; i < nObject; i++) {
    if (objects[i]->getNumList() > 2) return true;
  }

  return false;
}

__device__
BvhNode* BvhNode::build(Hittable **objects, int n, curandState* randState) {
  BvhNode *root = new BvhNode(objects, n);
  root->sortDivide(randState);

  BvhNode **leafNodes = (BvhNode**) malloc(n * sizeof(BvhNode*));
  int nLeaf = 0;

  leafNodes[nLeaf++] = root;

  while (isNotLeaf(leafNodes, nLeaf)) {
    int nCurNodes = nLeaf;

    for (int i = 0; i < nCurNodes; i++) {
      if (leafNodes[0]->getNumList() > 2) {
        BvhNode *leftNode = (BvhNode*) leafNodes[0]->getLeft();
        BvhNode *rightNode = (BvhNode*) leafNodes[0]->getRight();
        
        leafNodes[nLeaf++] = leftNode;
        leafNodes[nLeaf++] = rightNode;
      }

      for (int j = 1; j < nLeaf; j++) {
        leafNodes[j - 1] = leafNodes[j];
      }

      nLeaf--;
    }

    for (int i = 0; i < nLeaf; i++) {
      leafNodes[i]->sortDivide(randState);
    }
  }

  free(leafNodes);

  root->boundingBox(nullptr);
  return root;
}

__host__
BvhNode* BvhNode::build(Hittable **objects, int n) {
  BvhNode *root = new BvhNode(objects, n);
  root->sortDivide();

  BvhNode **leafNodes = (BvhNode**) malloc(n * sizeof(BvhNode*));
  int nLeaf = 0;

  leafNodes[nLeaf++] = root;

  while (isNotLeaf(leafNodes, nLeaf)) {
    int nCurNodes = nLeaf;

    for (int i = 0; i < nCurNodes; i++) {
      if (leafNodes[0]->getNumList() > 2) {
        BvhNode *leftNode = (BvhNode*) leafNodes[0]->getLeft();
        BvhNode *rightNode = (BvhNode*) leafNodes[0]->getRight();
        
        leafNodes[nLeaf++] = leftNode;
        leafNodes[nLeaf++] = rightNode;
      }

      for (int j = 1; j < nLeaf; j++) {
        leafNodes[j - 1] = leafNodes[j];
      }

      nLeaf--;
    }

    for (int i = 0; i < nLeaf; i++) {
      leafNodes[i]->sortDivide();
    }
  }

  free(leafNodes);

  root->boundingBox(nullptr);
  return root;
}

__host__ 
Hittable* BvhNode::copyToDevice() {
  Hittable *cudaLeftObject = this->leftObject->copyToDevice();

  if (this->leftObject != this->rightObject) {
    Hittable *cudaRightObject = this->rightObject->copyToDevice();
    this->rightObject = cudaRightObject;
  } else {
    this->rightObject = cudaLeftObject;
  }
  
  this->leftObject = cudaLeftObject;  

  BvhNode *cudaHit;

  cudaMalloc((void**) &cudaHit, sizeof(*this));
  cudaMemcpy(cudaHit, this, sizeof(*this), cudaMemcpyHostToDevice);

  return cudaHit;
}