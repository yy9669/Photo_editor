#include<opencv2/opencv.hpp>
#include<stdio.h>
#include<stdlib.h>
#include<vector>
#include<stack>
#include<cmath>
#include<time.h>
using namespace std;
using namespace cv;

Mat img0,img,tmp,Gra_show,gray,Gra,mask,tmask,pmask,tpmask;
Point pre_pt;
float MAX=1000000000;
short a[2000][2000];
int ss;

void calG(Mat &img, Mat&Gra){
	Mat Gx(img.rows,img.cols,CV_32F,Scalar(0));
	Mat Gy(img.rows,img.cols,CV_32F,Scalar(0));
	Mat kerx = (Mat_<float>(3,3) << 0, 0, 0, 0, 1, -1, 0, 0, 0);
	Mat kery = (Mat_<float>(3,3) << 0, 0, 0, 0, 1, 0, 0, -1, 0);
	filter2D(img,Gx,Gx.depth(),kerx);
	filter2D(img,Gy,Gy.depth(),kery);
	add(abs(Gx),abs(Gy),Gra);
}

void maskG(Mat &Gra, Mat &mask, Mat& pmask){
	int count = 0;
	srand((unsigned)time(0));
	//printf("%d %d\n",Gra.rows,mask.rows);
	for (int i=0;i<mask.rows;++i)
	for (int j=0;j<mask.cols;++j)
		if (mask.at<uchar>(i,j)==255){
			++count;
			//printf("%d %d\n",i,j);
			//Gra.at<float>(i,j) = -99999999;
		}
	int c0 = (count*rand())/(RAND_MAX+1);
	int c1 = (count*rand())/(RAND_MAX+1);
	int c2 = (count*rand())/(RAND_MAX+1);
	int c3 = (count*rand())/(RAND_MAX+1);
	int c4 = (count*rand())/(RAND_MAX+1);
	int c5 = (count*rand())/(RAND_MAX+1);
	int c6 = (count*rand())/(RAND_MAX+1);
	int c7 = (count*rand())/(RAND_MAX+1);
	int c8 = (count*rand())/(RAND_MAX+1);
	for (int i=0;i<mask.rows;++i)
	for (int j=0;j<mask.cols;++j)
		if (mask.at<uchar>(i,j)==255){
			--c0;--c1;--c2;--c3;--c4;--c5;--c6;--c7;--c8;
			if(c0==0||c1==0||c2==0||c3==0||c4==0||c5==0||c6==0||c7==0||c8==0) Gra.at<float>(i,j) = -99999999;
		}
	for (int i=0;i<pmask.rows;++i)
	for (int j=0;j<pmask.cols;++j)
		if (pmask.at<uchar>(i,j) == 255) Gra.at<float>(i,j) += 999999;
}

bool check(){
	int count=0;
	for (int i=0;i<mask.rows;++i)
	for (int j=0;j<mask.cols;++j)
		if (mask.at<uchar>(i,j)==255) ++count;
	if (count>20) return true;
	return false;
}

float dis(Mat &img,int x1, int y1, int x2, int y2){
	float a = img.at<Vec3b>(x1,y1)[0]-img.at<Vec3b>(x2,y2)[0];
	float b = img.at<Vec3b>(x1,y1)[1]-img.at<Vec3b>(x2,y2)[1];
	float c = img.at<Vec3b>(x1,y1)[2]-img.at<Vec3b>(x2,y2)[2];
	return sqrt(a*a+b*b+c*c);
}

vector<int> get_seam_x(Mat &img, float** energy, int row, int col, int k=1)
{
	float* dp = new float [row*col];
	int* flag = new int [row*col];
	for(int i = 0, j = 0; i < row; i++)
	{
		dp[i*col + j] = energy[i][j];
		flag[i*col + j] = i;
	}
	for(int j = 1; j < col; j++)
	{
		//for(int i = 0; i < row; i++)
		for (int i=1;i<row-1;++i)
		{
			int t = max(0, i-k);//t=i-1
			flag[i*col + j] = t;
			if(i-1<0||i+1>row-1){
				t=i;
				flag[i*col+j]=t;
				dp[i*col + j] = dp[t*col + j-1] + energy[i][j];
				if(i-1<0) dp[i*col + j]+=energy[i+1][j]+energy[i+1][j-1];
				else dp[i*col + j]+=energy[i-1][j]+energy[i-1][j-1];
			}
			else{
				float s[3]={0,0,0};
				if (i>1)s[0]=dis(img,i+1,j,i-1,j)+dis(img,i-1,j,i,j-1)+dp[(i-1)*col+j-1];
				//else *(int*)(&s[0]) = 0x7F800000;
			    s[1]=dis(img,i+1,j,i-1,j)+dp[i*col+j-1];
			    if (i<row-2) s[2]=dis(img,i+1,j,i-1,j)+dis(img,i+1,j,i,j-1)+dp[(i+1)*col+j-1];
				//else *(int*)(&s[2]) = 0x7F800000;
				if(i>1&&s[0]<s[1]&&(s[0]<s[2]||i==row-2)){
					flag[i*col+j]=i-1;
					dp[i*col+j]=s[0];
				}
				/*else if((s[1]<s[0]||i==1)&&(s[1]<s[2]||i==row-2)){
					flag[i*col+j]=i;
					dp[i*col+j]=s[1];
				}*/
				else if(i<row-2&&s[2]<s[1]&&(s[2]<s[0]||i==1)){
					flag[i*col+j]=i+1;
					dp[i*col+j]=s[2];
				}
				else{
					flag[i*col+j]=i;
					dp[i*col+j]=s[1];
				}
				dp[i*col+j]+=energy[i][j];
			/*
			for(int p = t+1; p <= min(i+k, row-1); p++)// p=i,p<=i+1 
			{
				if(dp[p*col + j-1] < dp[t*col + j-1])
				{
					t = p;
					flag[i*col + j] = t;
				}
			}
			dp[i*col + j] = dp[t*col + j-1] + energy[i][j];
			*/
			}
		}
	}
	//int t = 0;
	int t=1;
	//for(int i = 0; i < row; i++)
	for (int i=1;i<row-1;++i)
	{
		if(dp[i*col + col-1] <= dp[t*col + col-1]) t = i;
	}
	stack<int> stk;
	stk.push(t);
	//printf("%d\n",t);
	for(int i = col-1; i > 0; i--)
	{
		t = flag[t*col + i];
		stk.push(t);
	}
	vector<int> seam;
	while(!stk.empty())
	{
		seam.push_back(stk.top());
		stk.pop();
	}
	delete [] dp;
	delete [] flag;
	return seam;
}

vector<int> get_seam_y(Mat &img,float** energy, int row, int col, int k=1)
{
	float* dp = new float [row*col];
	int* flag = new int [row*col];
	for(int i = 0, j = 0; j < col; j++)
	{
		dp[i*col + j] = energy[i][j];
		flag[i*col + j] = j;
	}
	/*
	for(int i = 1; i < row; i++)
	{
		for(int j = 0; j < col; j++)
		{
			int t = max(0, j-k);
			flag[i*col + j] = t;
			for(int p = t+1; p <= min(j+k, col-1); p++)
			{
				if(dp[(i-1)*col + p] < dp[(i-1)*col + t])
				{
					t = p;
					flag[i*col + j] = t;
				}
			}
			dp[i*col + j] = dp[(i-1)*col + t] + energy[i][j];
		}
	}
	*/
	for(int i = 1; i < row; i++)
	{
		for(int j = 1; j < col-1; j++)
		{
			int t = max(0, j-k);//t=i-1
			flag[i*col + j] = t;

				float s[3]={0,0,0};
				if (j>1) s[0]=dis(img,i,j-1,i,j+1)+dis(img,i-1,j,i,j-1)+dp[(i-1)*col+j-1];
				//else *(int*)(&s[0]) = 0x7F800000;
			    s[1]=dis(img,i,j-1,i,j+1)+dp[(i-1)*col+j];
			    if (j<col-2) s[2]=dis(img,i,j+1,i,j-1)+dis(img,i,j+1,i-1,j)+dp[(i-1)*col+j+1];
				//else *(int*)(&s[2]) = 0x7F800000;
				if(j>1&&s[0]<s[1]&&(s[0]<s[2]||j==col-2)){
					flag[i*col+j]=j-1;
					dp[i*col+j]=s[0];
				}
				/*else if((s[1]<s[0]||i==1)&&(s[1]<s[2]||i==row-2)){
					flag[i*col+j]=i;
					dp[i*col+j]=s[1];
				}*/
				else if(j<col-2&&s[2]<s[1]&&(s[2]<s[0]||j==1)){
					flag[i*col+j]=j+1;
					dp[i*col+j]=s[2];
				}
				else{
					flag[i*col+j]=j;
					dp[i*col+j]=s[1];
				}
				dp[i*col+j]+=energy[i][j];
		}
	}
	int t = 1;
	for(int j = 1; j < col-1; j++)
	{
		if(dp[(row-1)*col + j] < dp[(row-1)*col + t]) t = j;
	}
	stack<int> stk;
	stk.push(t);
	for(int i = row-1; i > 0; i--)
	{
		t = flag[i*col + t];
		stk.push(t);
	}
	vector<int> seam;
	while(!stk.empty())
	{
		seam.push_back(stk.top());
		stk.pop();
	}
	delete [] dp;
	delete [] flag;
	return seam;
}


void cut_row(Mat &img, float **Gra_mat){
	tmp = Mat(img.rows-1,img.cols,img.type(),Scalar(0));
		vector<int> vec = get_seam_x(img,Gra_mat,Gra.rows,Gra.cols);
		for (int col=0;col<img.cols;++col){
			int row = vec[col];
			img.at<Vec3b>(row,col)[0] = 0;
			img.at<Vec3b>(row,col)[1] = 0;
			img.at<Vec3b>(row,col)[2] = 255;
		}
		imshow("graph",img);
		waitKey(2);
		for (int col=0;col<Gra.cols;++col){
			int row = vec[col];
			//printf("%d ",row);
			for (int i=0;i<row;++i){
				tmp.at<Vec3b>(i,col)[0] = img.at<Vec3b>(i,col)[0];
				tmp.at<Vec3b>(i,col)[1] = img.at<Vec3b>(i,col)[1];
				tmp.at<Vec3b>(i,col)[2] = img.at<Vec3b>(i,col)[2];
			}
			for (int i=row;i<img.rows-1;++i){
				tmp.at<Vec3b>(i,col)[0] = img.at<Vec3b>(i+1,col)[0];
				tmp.at<Vec3b>(i,col)[1] = img.at<Vec3b>(i+1,col)[1];
				tmp.at<Vec3b>(i,col)[2] = img.at<Vec3b>(i+1,col)[2];
			}
		}
		//printf("\n");
		//img = Mat(tmp.rows,tmp.cols,tmp.type(),Scalar(0));
		tmp.copyTo(img);
		gray = Mat(img.rows,img.cols,CV_8U,Scalar(0));
		Gra = Mat(img.rows,img.cols,CV_32F,Scalar(0));
		cvtColor(img,gray,COLOR_BGR2GRAY);
		calG(gray,Gra);
		for (int i=0;i<Gra.rows;++i)
			for (int j=0;j<Gra.cols;++j)  Gra_mat[i][j] = Gra.at<float>(i,j);
}

void cut_row3(Mat &img, float **Gra_mat){
	tmp = Mat(img.rows-1,img.cols,img.type(),Scalar(0));
	tmask = Mat(mask.rows-1,mask.cols,mask.type(),Scalar(0));
	tpmask = Mat(pmask.rows-1,pmask.cols,pmask.type(),Scalar(0));
		vector<int> vec = get_seam_x(img,Gra_mat,Gra.rows,Gra.cols);
		for (int col=0;col<img.cols;++col){
			int row = vec[col];
			img.at<Vec3b>(row,col)[0] = 0;
			img.at<Vec3b>(row,col)[1] = 0;
			img.at<Vec3b>(row,col)[2] = 255;
		}
		imshow("graph",img);
		waitKey(2);
		for (int col=0;col<Gra.cols;++col){
			int row = vec[col];
			for (int i=0;i<row;++i){
				tmp.at<Vec3b>(i,col)[0] = img.at<Vec3b>(i,col)[0];
				tmp.at<Vec3b>(i,col)[1] = img.at<Vec3b>(i,col)[1];
				tmp.at<Vec3b>(i,col)[2] = img.at<Vec3b>(i,col)[2];
				tmask.at<uchar>(i,col) = mask.at<uchar>(i,col);
				tpmask.at<uchar>(i,col) = pmask.at<uchar>(i,col);
			}
			for (int i=row;i<img.rows-1;++i){
				tmp.at<Vec3b>(i,col)[0] = img.at<Vec3b>(i+1,col)[0];
				tmp.at<Vec3b>(i,col)[1] = img.at<Vec3b>(i+1,col)[1];
				tmp.at<Vec3b>(i,col)[2] = img.at<Vec3b>(i+1,col)[2];
				tmask.at<uchar>(i,col) = mask.at<uchar>(i+1,col);
				tpmask.at<uchar>(i,col) = pmask.at<uchar>(i+1,col);
			}
		}
		tmp.copyTo(img);
		tmask.copyTo(mask);
		tpmask.copyTo(pmask);
		gray = Mat(img.rows,img.cols,CV_8U,Scalar(0));
		Gra = Mat(img.rows,img.cols,CV_32F,Scalar(0));
		cvtColor(img,gray,COLOR_BGR2GRAY);
		calG(gray,Gra);
		maskG(Gra,mask,pmask);
		for (int i=0;i<Gra.rows;++i)
			for (int j=0;j<Gra.cols;++j)  Gra_mat[i][j] = Gra.at<float>(i,j);
}

void cut_col(Mat &img, float **Gra_mat){
	tmp = Mat(img.rows,img.cols-1,img.type(),Scalar(0));
		vector<int> vec = get_seam_y(img,Gra_mat,Gra.rows,Gra.cols);
		for (int row=0;row<img.rows;++row){
			int col = vec[row];
			img.at<Vec3b>(row,col)[0] = 0;
			img.at<Vec3b>(row,col)[1] = 0;
			img.at<Vec3b>(row,col)[2] = 255;
		}
		imshow("graph",img);
		waitKey(2);
		for (int row=0;row<Gra.rows;++row){
			int col = vec[row];
			for (int i=0;i<col;++i){
				tmp.at<Vec3b>(row,i)[0] = img.at<Vec3b>(row,i)[0];
				tmp.at<Vec3b>(row,i)[1] = img.at<Vec3b>(row,i)[1];
				tmp.at<Vec3b>(row,i)[2] = img.at<Vec3b>(row,i)[2];
			}
			for (int i=col;i<img.cols-1;++i){
				tmp.at<Vec3b>(row,i)[0] = img.at<Vec3b>(row,i+1)[0];
				tmp.at<Vec3b>(row,i)[1] = img.at<Vec3b>(row,i+1)[1];
				tmp.at<Vec3b>(row,i)[2] = img.at<Vec3b>(row,i+1)[2];
			}
		}
		tmp.copyTo(img);
		gray = Mat(img.rows,img.cols,CV_8U,Scalar(0));
		Gra = Mat(img.rows,img.cols,CV_32F,Scalar(0));
		cvtColor(img,gray,COLOR_BGR2GRAY);
		calG(gray,Gra);
		for (int i=0;i<Gra.rows;++i)
			for (int j=0;j<Gra.cols;++j)  Gra_mat[i][j] = Gra.at<float>(i,j);
}

void cut_col3(Mat &img, float **Gra_mat){
	tmp = Mat(img.rows,img.cols-1,img.type(),Scalar(0));
	tmask = Mat(mask.rows,mask.cols-1,mask.type(),Scalar(0));
	tpmask = Mat(pmask.rows,pmask.cols-1,pmask.type(),Scalar(0));
		vector<int> vec = get_seam_y(img,Gra_mat,Gra.rows,Gra.cols);
		for (int row=0;row<img.rows;++row){
			int col = vec[row];
			img.at<Vec3b>(row,col)[0] = 0;
			img.at<Vec3b>(row,col)[1] = 0;
			img.at<Vec3b>(row,col)[2] = 255;
		}
		imshow("graph",img);
		waitKey(2);
		for (int row=0;row<Gra.rows;++row){
			int col = vec[row];
			for (int i=0;i<col;++i){
				tmp.at<Vec3b>(row,i)[0] = img.at<Vec3b>(row,i)[0];
				tmp.at<Vec3b>(row,i)[1] = img.at<Vec3b>(row,i)[1];
				tmp.at<Vec3b>(row,i)[2] = img.at<Vec3b>(row,i)[2];
				tmask.at<uchar>(row,i) = mask.at<uchar>(row,i);
				tpmask.at<uchar>(row,i) = pmask.at<uchar>(row,i);
			}
			for (int i=col;i<img.cols-1;++i){
				tmp.at<Vec3b>(row,i)[0] = img.at<Vec3b>(row,i+1)[0];
				tmp.at<Vec3b>(row,i)[1] = img.at<Vec3b>(row,i+1)[1];
				tmp.at<Vec3b>(row,i)[2] = img.at<Vec3b>(row,i+1)[2];
				tmask.at<uchar>(row,i) = mask.at<uchar>(row,i+1);
				tpmask.at<uchar>(row,i) = pmask.at<uchar>(row,i+1);
			}
		}
		tmp.copyTo(img);
		tmask.copyTo(mask);
		tpmask.copyTo(pmask);
		gray = Mat(img.rows,img.cols,CV_8U,Scalar(0));
		Gra = Mat(img.rows,img.cols,CV_32F,Scalar(0));
		cvtColor(img,gray,COLOR_BGR2GRAY);
		calG(gray,Gra);
		maskG(Gra,mask,pmask);
		for (int i=0;i<Gra.rows;++i)
			for (int j=0;j<Gra.cols;++j)  Gra_mat[i][j] = Gra.at<float>(i,j);
}

void expand_row(Mat &img, float **Gra_mat, int num1){
	img.copyTo(tmp);
	for(int i=0;i<=Gra.rows;i++){
				for(int j=0;j<=Gra.cols;j++) a[i][j]=0;
			}
			int n1=-num1;
			ss=n1/2;
			for(int t=1;t<=ss;t++){
			    vector<int> vec = get_seam_x(img,Gra_mat,Gra.rows,Gra.cols);
			    for(int i=0;i<Gra.cols;i++) {
					int row = vec[i];
					if(n1%ss>=t) a[row][i]=2;
					else 	a[row][i]=1;
					Gra_mat[row][i]=MAX;
					tmp.at<Vec3b>(row,i)[0] = 0;
					tmp.at<Vec3b>(row,i)[1] = 0;
					tmp.at<Vec3b>(row,i)[2] = 255;
				}
				imshow("graph",tmp);
				waitKey(2);
			}
			//tmp.copyTo(img);
			tmp = Mat(img.rows-num1,img.cols,img.type(),Scalar(0));
			for(int j=0;j<img.cols;j++){
				for(int i=0,ii=0;i<img.rows;i++,ii++){

					   tmp.at<Vec3b>(ii,j)[0] = img.at<Vec3b>(i,j)[0];
				       tmp.at<Vec3b>(ii,j)[1] = img.at<Vec3b>(i,j)[1];
				       tmp.at<Vec3b>(ii,j)[2] = img.at<Vec3b>(i,j)[2];
					if(a[i][j]>0){
						for(int p=0;p<n1/ss;p++){
						   ii++;
					       tmp.at<Vec3b>(ii,j)[0] = img.at<Vec3b>(i,j)[0];
				           tmp.at<Vec3b>(ii,j)[1] = img.at<Vec3b>(i,j)[1];
				           tmp.at<Vec3b>(ii,j)[2] = img.at<Vec3b>(i,j)[2];
						}
						if(a[i][j]>1){
						   ii++;
					       tmp.at<Vec3b>(ii,j)[0] = img.at<Vec3b>(i,j)[0];
				           tmp.at<Vec3b>(ii,j)[1] = img.at<Vec3b>(i,j)[1];
				           tmp.at<Vec3b>(ii,j)[2] = img.at<Vec3b>(i,j)[2];
						}
					}
				}
			}
					tmp.copyTo(img);
					gray = Mat(img.rows,img.cols,CV_8U,Scalar(0));
		Gra = Mat(img.rows,img.cols,CV_32F,Scalar(0));
		cvtColor(img,gray,COLOR_BGR2GRAY);
		calG(gray,Gra);
		for (int i=0;i<Gra.rows;++i)
			for (int j=0;j<Gra.cols;++j)  Gra_mat[i][j] = Gra.at<float>(i,j);
}

void expand_col(Mat &img, float **Gra_mat, int num2){
	img.copyTo(tmp);
	for(int i=0;i<=Gra.rows;i++){
				for(int j=0;j<=Gra.cols;j++) a[i][j]=0;
			}
			int n2=-num2;
			ss=n2/2;
			for(int t=1;t<=ss;t++){
			    vector<int> vec = get_seam_y(img,Gra_mat,Gra.rows,Gra.cols);
			    for(int row=0;row<Gra.rows;row++) {
					int col = vec[row];
					if(n2%ss>=t) a[row][col]=2;
					else 	a[row][col]=1;
					Gra_mat[row][col]=MAX;
					tmp.at<Vec3b>(row,col)[0] = 0;
					tmp.at<Vec3b>(row,col)[1] = 0;
					tmp.at<Vec3b>(row,col)[2] = 255;
				}
				imshow("graph",tmp);
				waitKey(2);
			}
			//tmp.copyTo(img);
			tmp = Mat(img.rows,img.cols-num2,img.type(),Scalar(0));
			for(int j=0;j<img.rows;j++){
				for(int i=0,ii=0;i<img.cols;i++,ii++){

					   tmp.at<Vec3b>(j,ii)[0] = img.at<Vec3b>(j,i)[0];
				       tmp.at<Vec3b>(j,ii)[1] = img.at<Vec3b>(j,i)[1];
				       tmp.at<Vec3b>(j,ii)[2] = img.at<Vec3b>(j,i)[2];
					if(a[j][i]>0){
						for(int p=0;p<n2/ss;p++){
						   ii++;
					       tmp.at<Vec3b>(j,ii)[0] = img.at<Vec3b>(j,i)[0];
				           tmp.at<Vec3b>(j,ii)[1] = img.at<Vec3b>(j,i)[1];
				           tmp.at<Vec3b>(j,ii)[2] = img.at<Vec3b>(j,i)[2];
						}
						if(a[j][i]>1){
						   ii++;
					       tmp.at<Vec3b>(j,ii)[0] = img.at<Vec3b>(j,i)[0];
				           tmp.at<Vec3b>(j,ii)[1] = img.at<Vec3b>(j,i)[1];
				           tmp.at<Vec3b>(j,ii)[2] = img.at<Vec3b>(j,i)[2];
						}
					}
				}
			}
					tmp.copyTo(img);
					gray = Mat(img.rows,img.cols,CV_8U,Scalar(0));
		Gra = Mat(img.rows,img.cols,CV_32F,Scalar(0));
		cvtColor(img,gray,COLOR_BGR2GRAY);
		calG(gray,Gra);
		for (int i=0;i<Gra.rows;++i)
			for (int j=0;j<Gra.cols;++j)  Gra_mat[i][j] = Gra.at<float>(i,j);
}

void on_mouse(int ev, int x, int y, int flags, void *param){
	if(img.size==0) return;
	if( ev == EVENT_LBUTTONUP || !(flags & EVENT_FLAG_LBUTTON)) pre_pt = Point(-1,-1);
	else if( ev == EVENT_LBUTTONDOWN )  pre_pt = Point(x,y);
	else if( ev == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON) ) {
		Point pt = Point(x,y);
		if( pre_pt.x < 0 ) pre_pt = pt;
		line( mask, pre_pt, pt, Scalar(255), 5, 8, 0 );
		line( img, pre_pt, pt, Scalar(255), 5, 8, 0 ); 
		pre_pt = pt;
		imshow( "image", img ); 
	}
	else if(ev == EVENT_LBUTTONDBLCLK){
		floodFill(mask,Point(x,y),Scalar(255));
		imwrite("maskImg.bmp",mask);
	}
}

void on_mouse1(int ev, int x, int y, int flags, void *param){
	if(img.size==0) return;
	if( ev == EVENT_LBUTTONUP || !(flags & EVENT_FLAG_LBUTTON)) pre_pt = Point(-1,-1);
	else if( ev == EVENT_LBUTTONDOWN )  pre_pt = Point(x,y);
	else if( ev == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON) ) {
		Point pt = Point(x,y);
		if( pre_pt.x < 0 ) pre_pt = pt;
		line( pmask, pre_pt, pt, Scalar(255), 5, 8, 0 );
		line( img, pre_pt, pt, Scalar(255), 5, 8, 0 ); 
		pre_pt = pt;
		imshow( "image", img ); 
	}
	else if(ev == EVENT_LBUTTONDBLCLK){
		floodFill(pmask,Point(x,y),Scalar(255));
		imwrite("pmaskImg.bmp",pmask);
	}
}
void seam_carving_greedy(Mat &img, float **energy, int k = 1)
{
	int row = img.rows;
	int col = img.cols;
	float* dpx = new float [row*col];
	int* flagx = new int [row*col];
	for(int i = 0, j = 0; i < row; i++)
	{
		dpx[i*col + j] = energy[i][j];
		flagx[i*col + j] = i;
	}
	for(int j = 1; j < col; j++)
	{
		for (int i=1;i<row-1;++i)
		{
			int t = max(0, i-k);//t=i-1
			flagx[i*col + j] = t;
			if(i-1<0||i+1>row-1){
				t=i;
				flagx[i*col+j]=t;
				dpx[i*col + j] = dpx[t*col + j-1] + energy[i][j];
				if(i-1<0) dpx[i*col + j]+=energy[i+1][j]+energy[i+1][j-1];
				else dpx[i*col + j]+=energy[i-1][j]+energy[i-1][j-1];
			}
			else{
				float s[3]={0,0,0};
				if (i>1)s[0]=dis(img,i+1,j,i-1,j)+dis(img,i-1,j,i,j-1)+dpx[(i-1)*col+j-1];
			    s[1]=dis(img,i+1,j,i-1,j)+dpx[i*col+j-1];
			    if (i<row-2) s[2]=dis(img,i+1,j,i-1,j)+dis(img,i+1,j,i,j-1)+dpx[(i+1)*col+j-1];
				if(i>1&&s[0]<s[1]&&(s[0]<s[2]||i==row-2)){
					flagx[i*col+j]=i-1;
					dpx[i*col+j]=s[0];
				}
				else if(i<row-2&&s[2]<s[1]&&(s[2]<s[0]||i==1)){
					flagx[i*col+j]=i+1;
					dpx[i*col+j]=s[2];
				}
				else{
					flagx[i*col+j]=i;
					dpx[i*col+j]=s[1];
				}
				dpx[i*col+j]+=energy[i][j];
			}
		}
	}
	int tx=1;
	for (int i=1;i<row-1;++i)
	{
		if(dpx[i*col + col-1] <= dpx[tx*col + col-1]) tx = i;
	}
	float seamx = dpx[tx*col + col-1] / col;

	float* dpy = new float [row*col];
	int* flagy = new int [row*col];
	for(int i = 0, j = 0; j < col; j++)
	{
		dpy[i*col + j] = energy[i][j];
		flagy[i*col + j] = j;
	}
	for(int i = 1; i < row; i++)
	{
		for(int j = 1; j < col-1; j++)
		{
			int t = max(0, j-k);//t=i-1
			flagy[i*col + j] = t;

				float s[3]={0,0,0};
				if (j>1) s[0]=dis(img,i,j-1,i,j+1)+dis(img,i-1,j,i,j-1)+dpy[(i-1)*col+j-1];
			    s[1]=dis(img,i,j-1,i,j+1)+dpy[(i-1)*col+j];
			    if (j<col-2) s[2]=dis(img,i,j+1,i,j-1)+dis(img,i,j+1,i-1,j)+dpy[(i-1)*col+j+1];
				if(j>1&&s[0]<s[1]&&(s[0]<s[2]||j==col-2)){
					flagy[i*col+j]=j-1;
					dpy[i*col+j]=s[0];
				}
				else if(j<col-2&&s[2]<s[1]&&(s[2]<s[0]||j==1)){
					flagy[i*col+j]=j+1;
					dpy[i*col+j]=s[2];
				}
				else{
					flagy[i*col+j]=j;
					dpy[i*col+j]=s[1];
				}
				dpy[i*col+j]+=energy[i][j];
		}
	}
	int ty = 1;
	for(int j = 1; j < col-1; j++)
	{
		if(dpy[(row-1)*col + j] < dpy[(row-1)*col + ty]) ty = j;
	}
	float seamy = dpy[(row-1)*col + ty] / row;

	if(seamx <= seamy ) // i do not know what to do if seamx == seamy
	{
		stack<int> stk;
		stk.push(tx);
		for(int i = col-1; i > 0; i--)
		{
			tx = flagx[tx*col + i];
			stk.push(tx);
		}
		vector<int> vec;
		while(!stk.empty())
		{
			vec.push_back(stk.top());
			stk.pop();
		}

		tmp = Mat(img.rows-1,img.cols,img.type(),Scalar(0));

		for (int c=0;c<col;++c){
			int r = vec[c];
			img.at<Vec3b>(r,c)[0] = 0;
			img.at<Vec3b>(r,c)[1] = 0;
			img.at<Vec3b>(r,c)[2] = 255;
		}
		imshow("graph",img);
		waitKey(2);
		for (int c=0;c<Gra.cols;++c){
			int r = vec[c];
			for (int i=0;i<r;++i){
				tmp.at<Vec3b>(i,c)[0] = img.at<Vec3b>(i,c)[0];
				tmp.at<Vec3b>(i,c)[1] = img.at<Vec3b>(i,c)[1];
				tmp.at<Vec3b>(i,c)[2] = img.at<Vec3b>(i,c)[2];
			}
			for (int i=r;i<row-1;++i){
				tmp.at<Vec3b>(i,c)[0] = img.at<Vec3b>(i+1,c)[0];
				tmp.at<Vec3b>(i,c)[1] = img.at<Vec3b>(i+1,c)[1];
				tmp.at<Vec3b>(i,c)[2] = img.at<Vec3b>(i+1,c)[2];
			}
		}
		tmp.copyTo(img);
		gray = Mat(img.rows,img.cols,CV_8U,Scalar(0));
		Gra = Mat(img.rows,img.cols,CV_32F,Scalar(0));
		cvtColor(img,gray,COLOR_BGR2GRAY);
		calG(gray,Gra);
		for (int i=0;i<Gra.rows;++i)
			for (int j=0;j<Gra.cols;++j)  energy[i][j] = Gra.at<float>(i,j);
	}
	else
	{
		stack<int> stk;
		stk.push(ty);
		for(int i = row-1; i > 0; i--)
		{
			ty = flagy[i*col + ty];
			stk.push(ty);
		}
		vector<int> vec;
		while(!stk.empty())
		{
			vec.push_back(stk.top());
			stk.pop();
		}
		
		tmp = Mat(img.rows,img.cols-1,img.type(),Scalar(0));
		for (int r=0;r<img.rows;++r){
			int c = vec[r];
			img.at<Vec3b>(r,c)[0] = 0;
			img.at<Vec3b>(r,c)[1] = 0;
			img.at<Vec3b>(r,c)[2] = 255;
		}
		imshow("graph",img);
		waitKey(2);
		for (int r=0;r<Gra.rows;++r){
			int c = vec[r];
			for (int i=0;i<c;++i){
				tmp.at<Vec3b>(r,i)[0] = img.at<Vec3b>(r,i)[0];
				tmp.at<Vec3b>(r,i)[1] = img.at<Vec3b>(r,i)[1];
				tmp.at<Vec3b>(r,i)[2] = img.at<Vec3b>(r,i)[2];
			}
			for (int i=c;i<img.cols-1;++i){
				tmp.at<Vec3b>(r,i)[0] = img.at<Vec3b>(r,i+1)[0];
				tmp.at<Vec3b>(r,i)[1] = img.at<Vec3b>(r,i+1)[1];
				tmp.at<Vec3b>(r,i)[2] = img.at<Vec3b>(r,i+1)[2];
			}
		}
		tmp.copyTo(img);
		gray = Mat(img.rows,img.cols,CV_8U,Scalar(0));
		Gra = Mat(img.rows,img.cols,CV_32F,Scalar(0));
		cvtColor(img,gray,COLOR_BGR2GRAY);
		calG(gray,Gra);
		for (int i=0;i<Gra.rows;++i)
			for (int j=0;j<Gra.cols;++j)  energy[i][j] = Gra.at<float>(i,j);
	}

	delete [] dpx;
	delete [] flagx;
	delete [] dpy;
	delete [] flagy;
}
int main(){
	printf("What do you want to do?\n1:carve 2:emphasize 3:remove\n");
	int choice;
	scanf("%d",&choice);
	if (choice==1){
	int num1,num2;
	scanf("%d%d",&num1,&num2);
	img0 = imread("kevin.jpg");
	namedWindow("original");
	imshow("original",img0);

	img0.copyTo(img);
	imshow("graph",img);
	gray = Mat(img.rows,img.cols,CV_8U,Scalar(0));
	Gra = Mat(img.rows,img.cols,CV_32F,Scalar(0));
	cvtColor(img,gray,COLOR_BGR2GRAY);
	calG(gray,Gra);
	Gra.convertTo(Gra_show,CV_8U,1,0);
	imshow("Gradient",Gra_show);
	waitKey();
	int tmp_rows = max(Gra.rows,Gra.rows-num1);
	int tmp_cols = max(Gra.cols,Gra.cols-num2);
	float **Gra_mat = new float*[tmp_rows];
		for (int i=0;i<tmp_rows;++i) Gra_mat[i] = new float [tmp_cols];
		for (int i=0;i<Gra.rows;++i)
			for (int j=0;j<Gra.cols;++j)  Gra_mat[i][j] = Gra.at<float>(i,j);
	if (num1<0) expand_row(img,Gra_mat,num1);
	else while (num1--){
		cut_row(img,Gra_mat);
		Gra.convertTo(Gra_show,CV_8U,1,0);
		imshow("Gradient",Gra_show);
		imshow("graph",img);
		waitKey(2);
	}
	if (num2<0) expand_col(img,Gra_mat,num2);
	else while (num2--){
		cut_col(img,Gra_mat);
		Gra.convertTo(Gra_show,CV_8U,1,0);
		imshow("Gradient",Gra_show);
		imshow("graph",img);
		waitKey(2);

	}
	imwrite("output.bmp",img);
	namedWindow("output");
	imshow("output",img);
	waitKey();
	}
	else if (choice == 2)
	{
		double ratio;
		cout << "please enter the enlargement ratio:" << endl;
		cin >> ratio;
		ratio = 1/ratio;

		img0 = imread("1.jpg");
		img0.copyTo(img);
		imshow("graph",img);

		Size dsize(img.cols, img.rows);
		Mat desimg;
		//Mat desimg(img);
		int row_ = ceil(img.rows*ratio);
		if(row_ > img.rows) row_ = img.rows;
		int col_ = ceil(img.cols*ratio);
		if(col_ > img.cols) col_ = img.cols;

		gray = Mat(img.rows,img.cols,CV_8U,Scalar(0));
		Gra = Mat(img.rows,img.cols,CV_32F,Scalar(0));
		cvtColor(img,gray,COLOR_BGR2GRAY);
		calG(gray,Gra);
		Gra.convertTo(Gra_show,CV_8U,1,0);
		imshow("Gradient",Gra_show);
		waitKey();
		int tmp_rows = Gra.rows;
		int tmp_cols = Gra.cols;
		float **Gra_mat = new float*[tmp_rows];
			for (int i=0;i<tmp_rows;++i) Gra_mat[i] = new float [tmp_cols];
				for (int i=0;i<Gra.rows;++i)
				for (int j=0;j<Gra.cols;++j)  Gra_mat[i][j] = Gra.at<float>(i,j);
		while(row_ < img.rows || col_ < img.cols)
		{
			if(row_ == img.rows)
			{
				cut_col(img,Gra_mat);
				Gra.convertTo(Gra_show,CV_8U,1,0);
				imshow("Gradient",Gra_show);
				imshow("graph",img);
				waitKey(2);
			}
			else if(col_ == img.cols)
			{
				cut_row(img,Gra_mat);
				Gra.convertTo(Gra_show,CV_8U,1,0);
				imshow("Gradient",Gra_show);
				imshow("graph",img);
				waitKey(2);
			}
			else
			{
				seam_carving_greedy(img, Gra_mat);
				Gra.convertTo(Gra_show,CV_8U,1,0);
				imshow("Gradient",Gra_show);
				imshow("graph",img);
				waitKey(2);
			}
		}
		
		resize(img, desimg, dsize);
		
		imwrite("output.bmp",desimg);
		namedWindow("output");
		imshow("output",desimg);
		waitKey();

	}
	else if(choice==3){
	printf("DO you want to retore the original size of the graph?\n1:Yes 2:No\n");
	int restore=2;
	scanf("%d",&restore);
	img0=imread("1.jpg");
	namedWindow("image");
	img = img0.clone();
	mask = Mat::zeros(img.rows,img.cols,CV_8UC1);
	pmask = Mat::zeros(img.rows,img.cols,CV_8UC1);
	imshow( "image", img ); 
	setMouseCallback( "image", on_mouse, 0 );
	waitKey();
	img = img0.clone();
	imshow("image",img);
	setMouseCallback("image",on_mouse1,0);
	waitKey();
	img = img0.clone();
	imshow("graph",img);
	gray = Mat(img.rows,img.cols,CV_8U,Scalar(0));
	Gra = Mat(img.rows,img.cols,CV_32F,Scalar(0));
	cvtColor(img,gray,COLOR_BGR2GRAY);
	calG(gray,Gra);

	Gra.convertTo(Gra_show,CV_8U,1,0);
	imshow("Gradient",Gra_show);
	waitKey();

	maskG(Gra,mask,pmask);
	int height = mask.size().height;
	int width = mask.size().width;
	int minr=10000,minc=10000,maxr=-1,maxc=-1;
	for (int i=0;i<height;++i)
	for (int j=0;j<width;++j)
		if (mask.at<uchar>(i,j)==255){
			minr = min(minr,i);
			maxr = max(maxr,i);
			minc = min(minc,j);
			maxc = max(maxc,j);
		}
	int lenr = maxr - minr;
	int lenc = maxc - minc;
	float **Gra_mat = new float*[Gra.rows];
		for (int i=0;i<Gra.rows;++i) Gra_mat[i] = new float [Gra.cols];
		for (int i=0;i<Gra.rows;++i)
			for (int j=0;j<Gra.cols;++j)  Gra_mat[i][j] = Gra.at<float>(i,j);

	Gra.convertTo(Gra_show,CV_8U,1,0);
	imshow("Gradient",Gra_show);
	waitKey();

	int cut_count = 0;

	if (lenr<=lenc){
		while (check()){
			++cut_count;
			cut_row3(img,Gra_mat);
			imshow("graph",img);
			Gra.convertTo(Gra_show,CV_8U,1,0);
			imshow("Gradient",Gra_show);
			waitKey(2);
		}
		if(restore==1) expand_row(img,Gra_mat,-cut_count);
	}
	else{
		while (check()){
			++cut_count;
			cut_col3(img,Gra_mat);
			imshow("graph",img);
			Gra.convertTo(Gra_show,CV_8U,1,0);
			imshow("Gradient",Gra_show);
			waitKey(2);
		}
		if(restore==1) expand_col(img,Gra_mat,-cut_count);
	}
	imwrite("output.bmp",img);
	namedWindow("output");
	imshow("output",img);
	waitKey();
	}
	return 0;
}