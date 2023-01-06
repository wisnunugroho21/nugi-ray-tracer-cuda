/*
Mat4.cpp
Written by Matthew Fisher

a 4x4 Mat4 structure.  Used very often for affine vector transformations.
*/

#include "arr4.cuh"
#include "math/arr3.cuh"

class Mat4 {
	public:
		__host__ __device__ Mat4() {}
		__host__ __device__ Mat4(const Arr3 &v0, const Arr3 &v1, const Arr3 &v2);
		__host__ __device__ Mat4(const Arr4 &v0, const Arr4 &v1, const Arr4 &v2, const Arr4 &v3);
		__host__ __device__ Mat4(const float a[4][4]);

		__host__ __device__ static Mat4 identity();

		__host__ __device__ Arr4 operator [](int i) const;

		__host__ __device__ Mat4& operator += (const Mat4 &mat);
		__host__ __device__ Mat4& operator += (const Arr4 &vec);
		__host__ __device__ Mat4& operator += (const float t);

		__host__ __device__ Mat4& operator -= (const Mat4 &mat);
		__host__ __device__ Mat4& operator -= (const Arr4 &vec);
		__host__ __device__ Mat4& operator -= (const float t);

		__host__ __device__ Mat4& operator *= (const float t);
		__host__ __device__ Mat4& operator /= (const float t);

		__host__ __device__ Mat4 inverse() const;
		__host__ __device__ Mat4 transpose() const;

	private:
		float entries[4][4];
};

__host__ __device__
Arr4 Mat4::operator [](int i) const {
	return Arr4(
		this->entries[i][0],
		this->entries[i][1],
		this->entries[i][2],
		this->entries[i][3]
	);
}

__host__ __device__
Mat4::Mat4(const Arr3 &V0, const Arr3 &V1, const Arr3 &V2) {
	this->entries[0][0] = V0.x();
	this->entries[0][1] = V0.y();
	this->entries[0][2] = V0.z();
	this->entries[0][3] = 0.0f;

	this->entries[1][0] = V1.x();
	this->entries[1][1] = V1.y();
	this->entries[1][2] = V1.z();
	this->entries[1][3] = 0.0f;

	this->entries[2][0] = V2.x();
	this->entries[2][1] = V2.y();
	this->entries[2][2] = V2.z();
	this->entries[2][3] = 0.0f;

	this->entries[3][0] = 0.0f;
	this->entries[3][1] = 0.0f;
	this->entries[3][2] = 0.0f;
	this->entries[3][3] = 1.0f;
}

__host__ __device__
Mat4::Mat4(const Arr4 &V0, const Arr4 &V1, const Arr4 &V2, const Arr4 &V3) {
	this->entries[0][0] = V0.x();
	this->entries[0][1] = V0.y();
	this->entries[0][2] = V0.z();
	this->entries[0][3] = V0.w();

	this->entries[1][0] = V1.x();
	this->entries[1][1] = V1.y();
	this->entries[1][2] = V1.z();
	this->entries[1][3] = V1.w();

	this->entries[2][0] = V2.x();
	this->entries[2][1] = V2.y();
	this->entries[2][2] = V2.z();
	this->entries[2][3] = V2.w();

	this->entries[3][0] = V3.x();
	this->entries[3][1] = V3.y();
	this->entries[3][2] = V3.z();
	this->entries[3][3] = V3.w();
}

__host__ __device__
Mat4::Mat4(const float a[4][4]) {
	this->entries[0][0] = a[0][0];
	this->entries[0][1] = a[0][1];
	this->entries[0][2] = a[0][2];
	this->entries[0][3] = a[0][3];

	this->entries[1][0] = a[1][0];
	this->entries[1][1] = a[1][1];
	this->entries[1][2] = a[1][2];
	this->entries[1][3] = a[1][3];

	this->entries[2][0] = a[2][0];
	this->entries[2][1] = a[2][1];
	this->entries[2][2] = a[2][2];
	this->entries[2][3] = a[2][3];

	this->entries[3][0] = a[3][0];
	this->entries[3][1] = a[3][1];
	this->entries[3][2] = a[3][2];
	this->entries[3][3] = a[3][3];
}

__host__ __device__ 
Mat4 Mat4::inverse() const {
	//
	// Inversion by Cramer's rule.  Code taken from an Intel publication
	//
	double result[4][4];
	double tmp[12]; /* temp array for pairs */
	double src[16]; /* array of transpose source matrix */
	double det; /* determinant */

	/* transpose matrix */
	for (int i = 0; i < 4; i++) {
		src[i + 0 ] = (*this)[i][0];
		src[i + 4 ] = (*this)[i][1];
		src[i + 8 ] = (*this)[i][2];
		src[i + 12] = (*this)[i][3];
	}

	/* calculate pairs for first 8 elements (cofactors) */
	tmp[0] = src[10] * src[15];
	tmp[1] = src[11] * src[14];
	tmp[2] = src[9] * src[15];
	tmp[3] = src[11] * src[13];
	tmp[4] = src[9] * src[14];
	tmp[5] = src[10] * src[13];
	tmp[6] = src[8] * src[15];
	tmp[7] = src[11] * src[12];
	tmp[8] = src[8] * src[14];
	tmp[9] = src[10] * src[12];
	tmp[10] = src[8] * src[13];
	tmp[11] = src[9] * src[12];

	/* calculate first 8 elements (cofactors) */
	result[0][0] = tmp[0]*src[5] + tmp[3]*src[6] + tmp[4]*src[7];
	result[0][0] -= tmp[1]*src[5] + tmp[2]*src[6] + tmp[5]*src[7];
	result[0][1] = tmp[1]*src[4] + tmp[6]*src[6] + tmp[9]*src[7];
	result[0][1] -= tmp[0]*src[4] + tmp[7]*src[6] + tmp[8]*src[7];
	result[0][2] = tmp[2]*src[4] + tmp[7]*src[5] + tmp[10]*src[7];
	result[0][2] -= tmp[3]*src[4] + tmp[6]*src[5] + tmp[11]*src[7];
	result[0][3] = tmp[5]*src[4] + tmp[8]*src[5] + tmp[11]*src[6];
	result[0][3] -= tmp[4]*src[4] + tmp[9]*src[5] + tmp[10]*src[6];
	result[1][0] = tmp[1]*src[1] + tmp[2]*src[2] + tmp[5]*src[3];
	result[1][0] -= tmp[0]*src[1] + tmp[3]*src[2] + tmp[4]*src[3];
	result[1][1] = tmp[0]*src[0] + tmp[7]*src[2] + tmp[8]*src[3];
	result[1][1] -= tmp[1]*src[0] + tmp[6]*src[2] + tmp[9]*src[3];
	result[1][2] = tmp[3]*src[0] + tmp[6]*src[1] + tmp[11]*src[3];
	result[1][2] -= tmp[2]*src[0] + tmp[7]*src[1] + tmp[10]*src[3];
	result[1][3] = tmp[4]*src[0] + tmp[9]*src[1] + tmp[10]*src[2];
	result[1][3] -= tmp[5]*src[0] + tmp[8]*src[1] + tmp[11]*src[2];

	/* calculate pairs for second 8 elements (cofactors) */
	tmp[0] = src[2]*src[7];
	tmp[1] = src[3]*src[6];
	tmp[2] = src[1]*src[7];
	tmp[3] = src[3]*src[5];
	tmp[4] = src[1]*src[6];
	tmp[5] = src[2]*src[5];

	tmp[6] = src[0]*src[7];
	tmp[7] = src[3]*src[4];
	tmp[8] = src[0]*src[6];
	tmp[9] = src[2]*src[4];
	tmp[10] = src[0]*src[5];
	tmp[11] = src[1]*src[4];

	/* calculate second 8 elements (cofactors) */
	result[2][0] = tmp[0]*src[13] + tmp[3]*src[14] + tmp[4]*src[15];
	result[2][0] -= tmp[1]*src[13] + tmp[2]*src[14] + tmp[5]*src[15];
	result[2][1] = tmp[1]*src[12] + tmp[6]*src[14] + tmp[9]*src[15];
	result[2][1] -= tmp[0]*src[12] + tmp[7]*src[14] + tmp[8]*src[15];
	result[2][2] = tmp[2]*src[12] + tmp[7]*src[13] + tmp[10]*src[15];
	result[2][2] -= tmp[3]*src[12] + tmp[6]*src[13] + tmp[11]*src[15];
	result[2][3] = tmp[5]*src[12] + tmp[8]*src[13] + tmp[11]*src[14];
	result[2][3] -= tmp[4]*src[12] + tmp[9]*src[13] + tmp[10]*src[14];
	result[3][0] = tmp[2]*src[10] + tmp[5]*src[11] + tmp[1]*src[9];
	result[3][0] -= tmp[4]*src[11] + tmp[0]*src[9] + tmp[3]*src[10];
	result[3][1] = tmp[8]*src[11] + tmp[0]*src[8] + tmp[7]*src[10];
	result[3][1] -= tmp[6]*src[10] + tmp[9]*src[11] + tmp[1]*src[8];
	result[3][2] = tmp[6]*src[9] + tmp[11]*src[11] + tmp[3]*src[8];
	result[3][2] -= tmp[10]*src[11] + tmp[2]*src[8] + tmp[7]*src[9];
	result[3][3] = tmp[10]*src[10] + tmp[4]*src[8] + tmp[9]*src[9];
	result[3][3] -= tmp[8]*src[9] + tmp[11]*src[10] + tmp[5]*src[8];

	/* calculate determinant */
	det=src[0]*result[0][0]+src[1]*result[0][1]+src[2]*result[0][2]+src[3]*result[0][3];
	/* calculate matrix inverse */
	det = 1.0f / det;

	float floatresult[4][4];
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			floatresult[i][j] = float(result[i][j] * det);
		}
	}

	return Mat4(floatresult);
}

__host__ __device__
Mat4 Mat4::transpose() const {
	float result[4][4];
	for(int i = 0; i < 4; i++) {
		for(int i2 = 0; i2 < 4; i2++) {
			result[i2][i] = this->entries[i][i2];
		}
	}

	return Mat4(
    Arr4(result[0][0], result[0][1], result[0][2], result[0][3]),
    Arr4(result[1][0], result[1][1], result[1][2], result[1][3]),
    Arr4(result[2][0], result[2][1], result[2][2], result[2][3]),
    Arr4(result[3][0], result[3][1], result[3][2], result[3][3])
  );
}

__host__ __device__
Mat4 Mat4::identity() {
	float result[4][4];
	for(int i = 0; i < 4; i++) {
		for(int i2 = 0; i2 < 4; i2++) {
			if(i == i2) {
				result[i][i2] = 1.0f;
			}
			
			else {
				result[i][i2] = 0.0f;
			}
		}
	}
	
	return Mat4(result);
}

__host__ __device__
Mat4& Mat4::operator += (const Mat4 &mat) {
	for(int i = 0; i < 4; i++) {
		for(int i2 = 0; i2 < 4; i2++) {
			this->entries[i][i2] = this->entries[i][i2] + mat[i][i2];
		}
	}

	return *this;
}

__host__ __device__
Mat4& Mat4::operator += (const Arr4 &vec) {
	for(int i = 0; i < 4; i++) {
		for(int i2 = 0; i2 < 4; i2++) {
			this->entries[i][i2] = this->entries[i][i2] + vec[i];
		}
	}

	return *this;
}

__host__ __device__
Mat4& Mat4::operator += (const float t) {
	for(int i = 0; i < 4; i++) {
		for(int i2 = 0; i2 < 4; i2++) {
			this->entries[i][i2] = this->entries[i][i2] + t;
		}
	}

	return *this;
}

__host__ __device__
Mat4& Mat4::operator -= (const Mat4 &mat) {
	for(int i = 0; i < 4; i++) {
		for(int i2 = 0; i2 < 4; i2++) {
			this->entries[i][i2] = this->entries[i][i2] - mat[i][i2];
		}
	}

	return *this;
}

__host__ __device__
Mat4& Mat4::operator -= (const Arr4 &vec) {
	for(int i = 0; i < 4; i++) {
		for(int i2 = 0; i2 < 4; i2++) {
			this->entries[i][i2] = this->entries[i][i2] - vec[i];
		}
	}

	return *this;
}

__host__ __device__
Mat4& Mat4::operator -= (const float t) {
	for(int i = 0; i < 4; i++) {
		for(int i2 = 0; i2 < 4; i2++) {
			this->entries[i][i2] = this->entries[i][i2] - t;
		}
	}

	return *this;
}

__host__ __device__
Mat4& Mat4::operator *= (const float t) {
	for(int i = 0; i < 4; i++) {
		for(int i2 = 0; i2 < 4; i2++) {
			this->entries[i][i2] = this->entries[i][i2] * t;
		}
	}

	return *this;
}

__host__ __device__
Mat4& Mat4::operator /= (const float t) {
	for(int i = 0; i < 4; i++) {
		for(int i2 = 0; i2 < 4; i2++) {
			this->entries[i][i2] = this->entries[i][i2] / t;
		}
	}

	return *this;
}

__host__ __device__
Mat4 operator * (const Mat4 &left, const Mat4 &right) {
	float result[4][4];
	for(int i = 0; i < 4; i++) {
		for(int i2 = 0; i2 < 4; i2++) {
			float Total = 0.0f;

			for(int i3 = 0; i3 < 4; i3++) {
				Total += left[i][i3] * right[i3][i2];
			}

			result[i][i2] = Total;
		}
	}

	return Mat4(result);
}

__host__ __device__
Arr4 operator * (const Mat4 &left, const Arr4 &right) {
	float result[4];
	for (int i = 0; i < 4; i++) {
		float total = 0.0f;

		for (int i2 = 0; i2 < 4; i2++) {
			total += left[i][i2] * right[i2];
		}

		result[i] = total;
	}

	return Arr4(result[0], result[1], result[2], result[3]);
}

__host__ __device__
Mat4 operator * (const Mat4 &left, float &right) {
	float result[4][4];
	for(int i = 0; i < 4; i++) {
		for(int i2 = 0; i2 < 4; i2++) {
			result[i][i2] = left[i][i2] * right;
		}
	}

	return Mat4(result);
}

__host__ __device__
Mat4 operator * (float &left, const Mat4 &right) {
	float result[4][4];
	for(int i = 0; i < 4; i++) {
		for(int i2 = 0; i2 < 4; i2++) {
			result[i][i2] = right[i][i2] * left;
		}
	}

	return Mat4(result);
}

__host__ __device__
Mat4 operator / (const Mat4 &left, float &right) {
	float result[4][4];
	for(int i = 0; i < 4; i++) {
		for(int i2 = 0; i2 < 4; i2++) {
			result[i][i2] = left[i][i2] * right;
		}
	}

	return Mat4(result);
}

__host__ __device__
Mat4 operator / (float &left, const Mat4 &right) {
	float result[4][4];
	for(int i = 0; i < 4; i++) {
		for(int i2 = 0; i2 < 4; i2++) {
			result[i][i2] = left / right[i][i2];
		}
	}

	return Mat4(result);
}

__host__ __device__
Mat4 operator + (const Mat4 &left, const Mat4 &right) {
	float result[4][4];
	for(int i = 0; i < 4; i++) {
		for(int i2 = 0; i2 < 4; i2++) {
			result[i][i2] = left[i][i2] + right[i][i2];
		}
	}

	return Mat4(result);
}

__host__ __device__
Mat4 operator + (const Mat4 &left, const Arr4 &right) {
	float result[4][4];
	for(int i = 0; i < 4; i++) {
		for(int i2 = 0; i2 < 4; i2++) {
			result[i][i2] = left[i][i2] + right[i];
		}
	}

	return Mat4(result);
}

__host__ __device__
Mat4 operator + (const Arr4 &left, const Mat4 &right) {
	float result[4][4];
	for(int i = 0; i < 4; i++) {
		for(int i2 = 0; i2 < 4; i2++) {
			result[i][i2] = left[i] + right[i][i2];
		}
	}

	return Mat4(result);
}

__host__ __device__
Mat4 operator + (const Mat4 &left, float right) {
	float result[4][4];
	for(int i = 0; i < 4; i++) {
		for(int i2 = 0; i2 < 4; i2++) {
			result[i][i2] = left[i][i2] + right;
		}
	}

	return Mat4(result);
}

__host__ __device__
Mat4 operator + (float left, const Mat4 &right) {
	float result[4][4];
	for(int i = 0; i < 4; i++) {
		for(int i2 = 0; i2 < 4; i2++) {
			result[i][i2] = left + right[i][i2];
		}
	}

	return Mat4(result);
}

__host__ __device__
Mat4 operator - (const Mat4 &left, const Mat4 &right) {
	float result[4][4];
	for(int i = 0; i < 4; i++) {
		for(int i2 = 0; i2 < 4; i2++) {
			result[i][i2] = left[i][i2] - right[i][i2];
		}
	}

	return Mat4(result);
}

__host__ __device__
Mat4 operator - (const Mat4 &left, const Arr4 &right) {
	float result[4][4];
	for(int i = 0; i < 4; i++) {
		for(int i2 = 0; i2 < 4; i2++) {
			result[i][i2] = left[i][i2] - right[i];
		}
	}

	return Mat4(result);
}

__host__ __device__
Mat4 operator - (const Arr4 &left, const Mat4 &right) {
	float result[4][4];
	for(int i = 0; i < 4; i++) {
		for(int i2 = 0; i2 < 4; i2++) {
			result[i][i2] = left[i] - right[i][i2];
		}
	}

	return Mat4(result);
}

__host__ __device__
Mat4 operator - (const Mat4 &left, const float &right) {
	float result[4][4];
	for(int i = 0; i < 4; i++) {
		for(int i2 = 0; i2 < 4; i2++) {
			result[i][i2] = left[i][i2] - right;
		}
	}

	return Mat4(result);
}

__host__ __device__
Mat4 operator - (const float &left, const Mat4 &right) {
	float result[4][4];
	for(int i = 0; i < 4; i++) {
		for(int i2 = 0; i2 < 4; i2++) {
			result[i][i2] = left - right[i][i2];
		}
	}

	return Mat4(result);
}