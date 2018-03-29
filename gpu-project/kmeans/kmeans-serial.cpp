#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>

#define N 40000
#define K 4
#define D 16
#define INF 10000
#define Th 0.000000001

using namespace std;

vector<vector<double> > load_point(ifstream& f);
vector<vector<double> > load_cluster(ifstream& f);
double point_d(vector<double>, vector<double>);


int main(){
	ifstream f,f2;
	//f.open("color100.txt",ios_base::in);
	//f.open("kmeans_1.4k.2d.4000",ios_base::in);
	//f.open("kmeans_2.4k.2d.4w",ios_base::in);
	f.open("kmeans_3.4k.16d.4w",ios_base::in);
	//f2.open("center.txt",ios_base::in);
	//f2.open("kmeans_1.center",ios_base::in);
	//f2.open("kmeans_2.center",ios_base::in);
	f2.open("kmeans_3.center",ios_base::in);
	vector<vector<double> > cluster(K, vector<double>(D));
	vector<vector<double> > new_cluster(K, vector<double>(D));
	vector<vector<double> > point(N, vector<double>(D));
	vector<int> category(N, 0);
	vector<int> count(K,0);
	int flag=0;
	
	point = load_point(f);
	cluster = load_cluster(f2);
	
	
	for(int i = 0; i<N;i++){
		for(int j =0; j<D;j++)
			cout<<point[i][j]<<" ";
		cout<<"\n";
	}
	
	for(int i = 0; i<K;i++){
		for(int j =0; j<D;j++)
			cout<<cluster[i][j]<<" ";
		cout<<"\n";
	}
	
	
	
	
	int iter = 0;
	//double delta = INF;
	while(flag == 0){
		//determine classes
		for(int i=0;i<N;i++){
			double distance = INF;
			for(int j=0;j<K;j++){
				double distance_temp;
				distance_temp = point_d(point[i],cluster[j]);
				if(distance>distance_temp){
					distance = distance_temp;
					category[i]=j;	
				}
			}	
		}
		
		//reset
		
		for(int i=0;i<K;i++)
			for(int j=0;j<D;j++)
				new_cluster[i][j]=0;
		for(int i=0;i<K;i++)
			count[i]=0;
		
		//compute new clusters
		for(int i=0;i<N;i++){
			int index = category[i];
			count[index]++;
			for(int j=0;j<D;j++){
				new_cluster[index][j]+=point[i][j];
			}
		}
		
		for(int i=0;i<K;i++)
			for(int j=0;j<D;j++)
				new_cluster[i][j]/=count[i];
			
		//compute delta
		flag = 1;
		for(int i=0;i<K;i++){
			if(point_d(new_cluster[i],cluster[i])>Th)
				flag = 0;
		}
		
		//set cluster
		cluster = new_cluster;
		cout<<"OK!";
		iter++;
		cout<<"iter:"<<iter<<"\n";
		for(int i = 0; i<N;i++){
		for(int j =0; j<D;j++)
			//cout<<point[i][j]<<" ";
		//cout<<"\n";
		;
		}
	}
	
	for(int i = 0; i<K;i++){
		for(int j =0; j<D;j++)
			cout<<cluster[i][j]<<" ";
		cout<<"\n";
	}
	
	for(int i=0;i<N;i++)
		cout<<category[i]<<" ";
	cout<<"\n";
	
	for(int i=0;i<N;i++){
		for(int j=0;j<K;j++)
			cout<<point_d(point[i],cluster[j])<<" ";
		cout<<"\n";
	}
}

vector<vector<double> > load_point(ifstream& f){
	vector<vector<double> > point(N, vector<double>(D));
	for(int i=0;i<N;i++)
		for(int j=0;j<D;j++)
			f>>point[i][j];
	return point;
}

vector<vector<double> > load_cluster(ifstream& f){
	vector<vector<double> > cluster(K, vector<double>(D));
	for(int i=0;i<K;i++)
		for(int j=0;j<D;j++)
			f>>cluster[i][j];
	return cluster;
}

double point_d(vector<double> a, vector<double> b){
	double temp = 0.0;
	for(int i=0;i<a.size();i++){
		temp+= pow(a[i]-b[i],2);
		
	}
	
	return sqrt(temp);
}
