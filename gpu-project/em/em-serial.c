//
//  main.c
//  em serial
//
//  Created by Xi on 5/5/15.
//  Copyright (c) 2015 daphne. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define M 4000
#define K 4
#define N 2

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main()
{
    FILE *fp;// = fopen("mu.txt","r");
    //double mu[6];
    int i,j,iter,k;
    
    //double mu[K][N];
    double** mu = (double**)malloc(sizeof(double*)*K);
    for(i = 0; i< K; i++)
        mu[i] = (double*)malloc(sizeof(double)*N);
    
    //double prevMu[K][N];
    
    double** prevMu = (double**)malloc(sizeof(double*)*K);
    for(i = 0; i< K; i++)
        prevMu[i] = (double*)malloc(sizeof(double)*N);
    
    //double sigma[K][N][N];
    double*** sigma = (double***)malloc(sizeof(double**)*K);
    for(i = 0; i< K; i++){
        sigma[i] = (double**)malloc(sizeof(double*)*N);
        for(j=0;j<N;j++)
            sigma[i][j] = (double*)malloc(sizeof(double)*N);
    }
    
    
    
    //double sigma_inv[K][N][N];
    double*** sigma_inv = (double***)malloc(sizeof(double**)*K);
    for(i = 0; i< K; i++){
        sigma_inv[i] = (double**)malloc(sizeof(double*)*N);
        for(j=0;j<N;j++)
            sigma_inv[i][j] = (double*)malloc(sizeof(double)*N);
    }

    
    
    
    //double sigma_k[K][N][N];
    double*** sigma_k = (double***)malloc(sizeof(double**)*K);
    for(i = 0; i< K; i++){
        sigma_k[i] = (double**)malloc(sizeof(double*)*N);
        for(j=0;j<N;j++)
            sigma_k[i][j] = (double*)malloc(sizeof(double)*N);
    }

    
    
    //double X[M][N];
    double** X = (double**)malloc(sizeof(double*)*M);
    for(i = 0; i< M; i++)
        X[i] = (double*)malloc(sizeof(double)*N);
    
    //double Xm[K][M][N];
    double*** Xm = (double***)malloc(sizeof(double**)*K);
    for(i = 0; i< K; i++){
        Xm[i] = (double**)malloc(sizeof(double*)*M);
        for(j=0;j<M;j++)
            Xm[i][j] = (double*)malloc(sizeof(double)*N);
    }
    
    //double phi[K]={1/((double)K),1/((double)K)};
    double phi[K];
    for(i = 0; i<K; i++){
		phi[i]= 1/((double)(K)+0.00000001);
	}
    
    //double W[M][K];
    double** W = (double**)malloc(sizeof(double*)*M);
    for(i = 0; i< M; i++){
        W[i] = (double*)malloc(sizeof(double)*K);
    }
    
    //double pdf[M][K];
    double** pdf = (double**)malloc(sizeof(double*)*M);
    for(i = 0; i< M; i++){
        pdf[i] = (double*)malloc(sizeof(double)*K);
    }
    
    //double meanDiff[K][M][N];
    double*** meanDiff = (double***)malloc(sizeof(double**)*K);
    for(i = 0; i< K; i++){
        meanDiff[i] = (double**)malloc(sizeof(double*)*M);
        for(j=0;j<M;j++){
            meanDiff[i][j] = (double*)malloc(sizeof(double)*N);
		}
	}

    
    //double det_sig[K];
    double* det_sig = (double*)malloc(sizeof(double)*K);
    
    
    //double mean_inv[K][M][N];
    double*** mean_inv = (double***)malloc(sizeof(double**)*K);
    for(i = 0; i< K; i++){
        mean_inv[i] = (double**)malloc(sizeof(double*)*M);
        for(j=0;j<M;j++)
            mean_inv[i][j] = (double*)malloc(sizeof(double)*N);
    }

    
    //double mean_inv_mean[K][M][N];
    double*** mean_inv_mean = (double***)malloc(sizeof(double**)*K);
    for(i = 0; i< K; i++){
        mean_inv_mean[i] = (double**)malloc(sizeof(double*)*M);
        for(j=0;j<M;j++)
            mean_inv_mean[i][j] = (double*)malloc(sizeof(double)*N);
    }

    
    double sum_mean[K][M][1];
    
    //double con_pdf[K];
    double* con_pdf = (double*)malloc(sizeof(double)*K);
    
    //double pdf_w[M][K];
    double** pdf_w = (double**)malloc(sizeof(double*)*M);
    for(i = 0; i< M; i++)
        pdf_w[i] = (double*)malloc(sizeof(double)*K);

    
    double sum_pdfw[M][1];
    
    //double sum_W[K];
    double* sum_W = (double*)malloc(sizeof(double)*K);
    
    //double mu_sum[K][N];
    double** mu_sum = (double**)malloc(sizeof(double*)*K);
    for(i = 0; i< K; i++)
        mu_sum[i] = (double*)malloc(sizeof(double)*N);

    
    //printf("%lf\n",phi[0]);
   
    int flag=0;
    
    
    //if((fp=fopen("mu.txt","r"))!=NULL){
    //if((fp=fopen("em_1.mu","r"))!=NULL){
    if((fp=fopen("em_3.mu","r"))!=NULL){
        //fscanf(fp,"%f",&mu[0]);
        for(i=0;i<K;i++){
            for(j=0;j<N;j++)
               fscanf(fp,"%lf",&mu[i][j]);
        }
        fclose(fp);
    }
    
    //printf("%lf\n",mu[0][0]);
    //printf("%lf\n",mu[0][1]);
   //if((fp=fopen("covx.txt","r"))!=NULL){
   //if((fp=fopen("em_1.covx","r"))!=NULL){
   if((fp=fopen("em_3.covx","r"))!=NULL){
        //fscanf(fp,"%f",&mu[0]);
        for(k=0;k<K;k++){
            for(i=0;i<N;i++){
                for(j=0;j<N;j++)
                    fscanf(fp,"%lf",&sigma[k][i][j]);
            }
        }
        fclose(fp);
    }
    
    /*for(k=0;k<K;k++){
        sigma[k][0][0]=4;
        sigma[k][0][1]=3;
        sigma[k][1][0]=1;
        sigma[k][1][1]=1;
    }*/
    
   
    /*for(k=0;k<K;k++){
         for(i=0;i<N;i++){
            
                 printf("%lf %lf\n",sigma[k][i][0],sigma[k][i][1]);
         }
    }*/
    
    //if((fp=fopen("data.txt","r"))!=NULL){
    //if((fp=fopen("em_1.2k.10000","r"))!=NULL){
    if((fp=fopen("em_3.4k.4000","r"))!=NULL){
        //fscanf(fp,"%f",&mu[0]);
        for(i=0;i<M;i++){
            for(j=0;j<N;j++)
                fscanf(fp,"%lf",&X[i][j]);
        }
        fclose(fp);
    }
    
    
    //for(iter=0;iter<1000;iter++){
    for(iter=0;iter<10000;iter++){
        printf("EM Iteration %d\n",iter);
        
        for(k=0;k<K;k++){  //subtract the mean from every data point
            for(i=0;i<M;i++){
                for(j=0;j<N;j++){
                     meanDiff[k][i][j]=X[i][j]-mu[k][j];
                }
            }
        }
       // printf("%lf %lf\n\n",mu[0][0],mu[0][1]);
        for(i=0;i<1000;i++){
         // printf("%lf %lf\n",meanDiff[0][i][0],meanDiff[0][i][1]);
         }
    
        
        for(k=0;k<K;k++){//calculate the inverse of sigma
            det_sig[k]=sigma[k][0][0]*sigma[k][1][1]-sigma[k][0][1]*sigma[k][1][0];
            sigma_inv[k][0][0]=sigma[k][1][1]/det_sig[k];
            sigma_inv[k][0][1]=-sigma[k][0][1]/det_sig[k];
            sigma_inv[k][1][0]=-sigma[k][1][0]/det_sig[k];
            sigma_inv[k][1][1]=sigma[k][0][0]/det_sig[k];
            //printf("%lf\n",det_sig[k]);
            //printf("%lf %lf\n",sigma_inv[k][0][0],sigma_inv[k][0][1]);
            //printf("%lf %lf\n",sigma_inv[k][1][0],sigma_inv[k][1][1]);
        }
        
        for(k=0;k<K;k++){
            for(i=0;i<M;i++){
                for(j=0;j<N;j++){
                    mean_inv[k][i][j]=meanDiff[k][i][0]*sigma_inv[k][0][j]+meanDiff[k][i][1]*sigma_inv[k][1][j];
                }
            }
        }
        
        /*for(i=0;i<M;i++){
            
                printf("%lf %lf\n",mean_inv[0][i][0],mean_inv[0][i][1]);
           
        }*/
        
        for(k=0;k<K;k++){
            for(i=0;i<M;i++){
                for(j=0;j<N;j++){
                    mean_inv_mean[k][i][j]=mean_inv[k][i][j]*meanDiff[k][i][j];
                }
                sum_mean[k][i][0]=mean_inv_mean[k][i][0]+mean_inv_mean[k][i][1];
                sum_mean[k][i][0]=exp((-1/2.0)*sum_mean[k][i][0]);
                
            }
        }
        
        for(k=0;k<K;k++){
            con_pdf[k]=1/sqrt((pow(2*M_PI,N))*det_sig[k]);
            //printf("%lf\n",con_pdf[k]);
        }
        
        for(k=0;k<K;k++){
            for(i=0;i<M;i++){
                    pdf[i][k]=con_pdf[k]*sum_mean[k][i][0];
                    pdf_w[i][k]=pdf[i][k]*phi[k];
                    //printf("%lf\n",phi[k]);
            }
        }
        /*for(i=0;i<M;i++){
            
                 printf("%lf %lf %lf\n",pdf_w[i][0],pdf_w[i][1],pdf_w[i][2]);
            
        }*/
        for(i=0;i<M;i++){
            sum_pdfw[i][0]=pdf_w[i][0]+pdf_w[i][1]+pdf_w[i][2];
            for(k=0;k<K;k++){
                W[i][k]=pdf_w[i][k]/sum_pdfw[i][0];
            }
        }
        
        for(i=0;i<K;i++){
            for(j=0;j<N;j++)
                prevMu[i][j]=mu[i][j];
        }
        
        
        for(k=0;k<K;k++){
            sum_W[k]=0.0;
            for(i=0;i<M;i++){
                sum_W[k]=W[i][k]+sum_W[k];
            }
            phi[k]=sum_W[k]/M;
        }
        
        for(k=0;k<K;k++){
            for(j=0;j<N;j++){
                mu_sum[k][j]=0.0;
            }
            for(i=0;i<M;i++){
                for(j=0;j<N;j++)
                    mu_sum[k][j]=W[i][k]*X[i][j]+mu_sum[k][j];
            }
        }
        
        for(k=0;k<K;k++){
            for(j=0;j<N;j++){
                mu[k][j]=mu_sum[k][j]/sum_W[k];
            }
        }
        
        for(k=0;k<K;k++){
            printf("%lf %lf\n",mu[k][0],mu[k][1]);
        }
        for(k=0;k<K;k++){
            for(i=0;i<M;i++){
                for(j=0;j<N;j++){
                    Xm[k][i][j]=X[i][j]-mu[k][j];
					if(k == 0){
						// printf("%lf\n",Xm[k][i][j]);
					}
                }
            }
        }
        
        for(k=0;k<K;k++){
            for(i=0;i<N;i++){
                for(j=0;j<N;j++)
                    sigma_k[k][i][j]=0.0;
            }
        }
        
        for(k=0;k<K;k++){
            for(i=0;i<M;i++){
                sigma_k[k][0][0]=sigma_k[k][0][0]+Xm[k][i][0]*Xm[k][i][0]*W[i][k];
                sigma_k[k][0][1]=sigma_k[k][0][1]+Xm[k][i][0]*Xm[k][i][1]*W[i][k];
                sigma_k[k][1][0]=sigma_k[k][1][0]+Xm[k][i][0]*Xm[k][i][1]*W[i][k];
                sigma_k[k][1][1]=sigma_k[k][1][1]+Xm[k][i][1]*Xm[k][i][1]*W[i][k];
            }
        }
        
        
        for(k=0;k<K;k++){
            for(i=0;i<N;i++){
                for(j=0;j<N;j++)
                   sigma[k][i][j]=sigma_k[k][i][j]/sum_W[k];
            }
        }
        
        /*for(k=0;k<K;k++){
            for(i=0;i<N;i++){
                printf("%lf %lf\n",sigma[k][i][0],sigma[k][i][1]);
            }
        }*/
        flag=0;
    
        for(i=0;i<K;i++){
            for(j=0;j<N;j++){
                if (mu[i][j]!=prevMu[i][j])
                    flag=1;
            }
        }
    
        if(flag==0){
            break;
        }
    
        
    }
    
   
    
    
    
    
   /* for(k=0;k<K;k++){
        printf("%lf %lf\n",mu[k][0],mu[k][1]);
    }*/
    
    
    free(mu);
    
    free(prevMu);
    
    free(sigma);
    
    
    
    
    free(sigma_inv);
    
    
    free(sigma_k);
    
    
    
    
    free(X);
    
    free(Xm);
    free(W);
    
    free(pdf);
    
    
    free(meanDiff);

    
    free(det_sig);
    
    
    free(mean_inv);
    
    free(mean_inv_mean);
    
    free(con_pdf);
    
    free(pdf_w);
    
    free(sum_W);
    
    free(mu_sum);
    
    
    
    
    return 0;
}

