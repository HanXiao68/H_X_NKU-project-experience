#include "sys.h"
#include "delay.h"
#include "usart.h"
#include "led.h"
#include "beep.h"
#include "dac.h"
#include "pwm.h"
#include "timer.h"
   
void selfcheck(void);			//×Ô¼ìº¯ÊıÉùÃ÷

//ALIENTEK Ì½Ë÷ÕßSTM32F407¿ª·¢°å ÊµÑé4
//´®¿ÚÍ¨ĞÅÊµÑé -¿âº¯Êı°æ±¾
//¼¼ÊõÖ§³Ö£ºwww.openedv.com
//ÌÔ±¦µêÆÌ£ºhttp://eboard.taobao.com
//¹ãÖİÊĞĞÇÒíµç×Ó¿Æ¼¼ÓĞÏŞ¹«Ë¾  
//×÷Õß£ºÕıµãÔ­×Ó @ALIENTEK


/********************×¢ÊÍ********************

ËÙ¶È£ºcarSpeed=0~4000¡ª¡ª0~3.3V
     Æô¶¯Öµ£º1700--6A4
	   ×î´óÖµ£º4000--FA0
	  
×ªËÙ£ºarr
      ×îĞ¡Öµ£ºarr=4000--FA0  --250Hz
	  
temp160£ºbit0~4--×ª½ÇÖµ
         bit5  --×ª½Ç·ûºÅÎ»
		     bit6  --×ªÏò¸´Î»
		     bit7  --É²³µ
		     bit8  --×Ô¼ì
		     bit9  --µÍËÙ
		     bit10 --¸ßËÙ
		     bit11 --¶ÏÊ¹ÄÜ
				 bit12~15--¿Õ
		 
temp161£ºbit0~11--ËÙ¶ÈÖµ
         bit12  --Æô¶¯
		     bit13  --Ç°½ø/ºóÍË
		     bit14  --À®°È
		     bit15  --Ç°´óµÆ
		 
temp162£ºbit0~11--×ªËÙÖµ
         bit12  --ÓÒÉÁ
	    	 bit13  --×óÉÁ
	    	 bit14  --ÓÒ×ª
		     bit15  --×ó×ª

********************************************/




/****************²âÊÔÓÃÖ¸Áî******************

Ç°´óµÆ£º00 00 00 80 00 00				
×óÉÁµÆ£º00 20 00 00 00 00
ÓÒÉÁµÆ£º00 10 00 00 00 10
ÃùµÑ  £º00 00 00 40 00 00
Ç°½ø  £º00 00 00 10 00 00     £¨Ç°½ø»òÆô¶¯£©1800--708  2000-7D0  ËÙ¶ÈÖµ
µ¹ÍË  £º00 00 08 37 00 00		
É²³µ  £º00 00 00 00 80 00
ÓÒ×ª  £º00 40 00 00 00 00
×ó×ª  £º00 80 00 00 00 00
¸´Î»  £º00 00 00 00 40 00		£¿£¿£¿×ªÏò¸´Î»
×Ô¼ì  £º00 00 00 00 00 01   	
********************************************/





/********************************************

16½øÖÆ×ª10½øÖÆº¯Êı

·µ»ØÖµ£º10½øÖÆÊı

********************************************/

int Hex_Dec(u16 hex)
{
	u8 i;
	u16 a=0;
	int sum=0;
	
	for(i=0;i<3;++i)
	{			
	    a=hex&0x000f;
	    hex=hex>>4;
		switch(a)
		{
			case 0xa:a=10;break;
			case 0xb:a=11;break;
			case 0xc:a=12;break;
      case 0xd:a=13;break;
      case 0xe:a=14;break;
      case 0xf:a=15;break;
		}
		if(i==0)
			sum=a*1;
		if(i==1)
			sum=sum+a*16;
		if(i==2)
			sum=sum+a*16*16;
	}
	return sum;
}

/****************×Ô¼ìº¯Êı***********************/
void selfcheck(){		                                              			
			    printf("\r\n×Ô¼ìÆô¶¯\r\n");  
				
				LED1=!LED1;
		        delay_ms(2000);
				printf("\r\nÇ°´óµÆOK\r\n");
			    LED1=!LED1;
			    delay_ms(2000);
				
				BEEP=!BEEP;
			    delay_ms(500);
			    printf("\r\nÃùµÑOK\r\n");
			     BEEP=!BEEP;
			     delay_ms(2000);
				
			    LED2=!LED2;
		        delay_ms(2000);
				printf("\r\nÓÒÉÁµÆOK\r\n");
			    LED2=!LED2;
			    delay_ms(2000); 
								
				DAC_SetChannel1Data(DAC_Align_12b_R, 2000);
				printf("\r\nÇ°½øOK\r\n");
				Output_Pulse(1800);
				printf("\r\nÓÒ×ªOK\r\n");
				delay_ms(8000); 

//				delay_ms(4000); 

				SHACHE=!SHACHE;
				DAC_SetChannel1Data(DAC_Align_12b_R, 0);
				printf("\r\nÉ²³µOK\r\n");
				delay_ms(500);
				SHACHE=!SHACHE;
				delay_ms(1000);		
				
			    LED3=!LED3;
		        delay_ms(4000);
				printf("\r\n×óÉÁµÆOK\r\n");
			    LED3=!LED3;
			    delay_ms(2000);
				
 			    DAOCHE=!DAOCHE;
			 	delay_ms(500);
			    DAC_SetChannel1Data(DAC_Align_12b_R, 2000);
				PA6=!PA6;
				Output_Pulse(18100);
				printf("\r\nµ¹³µOK\r\n");
				delay_ms(8000); 
				
			
				printf("\r\n×ó×ªOK\r\n");
//				delay_ms(10000);
				
				SHACHE=!SHACHE;
				DAC_SetChannel1Data(DAC_Align_12b_R, 0);
			    delay_ms(500);
			    SHACHE=!SHACHE;
				delay_ms(500);
				DAOCHE=!DAOCHE;
				printf("\r\n×Ô¼ì½áÊø\r\n");			
}


/********************Ö÷º¯Êı********************/

int main(void)
{ 
	/********±äÁ¿¶¨Òå********/
	
	u8 t=0;
	u8 len=0;	
	u32 i=0;
	u16 res=0;
  u16 temp=0;
	u16 temp160=0;
	u16 temp161=0;
	u16 temp162=0;
	u16 carSpeed=0;
	u16 angel_hex=0;
	u16 rot_speed_hext=0;
	int angel=0;
	int Rangel=0;
	int Langel=0;
	int angel_re=0;
	int rot_speed=0;
	uint8_t statue1=0;
	uint8_t statue2=0;
	uint8_t statue3=0;
	uint8_t statue5=0;
	uint8_t statue6=0;
	uint8_t statue7=0;
	uint8_t statue8=0;
	uint8_t statue9=0;
/*********************³õÊ¼»¯***********************/
	
	
	NVIC_PriorityGroupConfig(NVIC_PriorityGroup_2);   //ÉèÖÃÏµÍ³ÖĞ¶ÏÓÅÏÈ¼¶·Ö×é2
	delay_init(168);		                          //ÑÓÊ±³õÊ¼»¯ 
	uart_init(115200);	                              //´®¿Ú³õÊ¼»¯²¨ÌØÂÊÎª115200
	
	BEEP_Init();
 	QianDaDeng_Init();                                //³õÊ¼»¯Ç°´óµÆ £¿£¿£¿£¿£¿À®°È£¿£¿
	ZuoShanDeng_Init();                               //³õÊ¼»¯×óÉÁµÆ
	YouShanDeng_Init();                               //³õÊ¼»¯ÓÒÉÁµÆ
 	ShaChe_Init();                                    //³õÊ¼»¯É²³µ
	DaoChe_Init();                                    //³õÊ¼»¯µ¹³µ
	GaoSu_Init();                                     //³õÊ¼»¯¸ßËÙ
	DiSu_Init();                                      //³õÊ¼»¯µÍËÙ
	HongWai_Init();
	Dac1_Init();                                      //³õÊ¼»¯DAC
	NVIC_Config();  					
	GPIO_Config();						//ÖĞ¶ÏÏòÁ¿Ç¶Ì×¿ØÖÆÆ÷£¬ÓÃÀ´¹ÜÀíËùÓĞÖÕ¶ËºÍÊÂ¼şµÄ¡££¨°üÀ¨ÖĞ¶ÏµÄÊ¹ÄÜºÍ³ıÄÜ£¬ÖĞ¶ÏµÄÓÅÏÈ¼¶£©©
	GPIO_EN_DIR_Init();				//ÔÚGPIOÄ£Ê½ÏÂ£¬Ã¿¸öIO¶¼¿ÉÅäÖÃÎªÊäÈë»òÊä³ö
	TIM2_Master__TIM3_Slave_Configuration(2250);	// ÅäÖÃTIM2µÄÂö³åÊä³öÎª10Hz  £¿£¿£¿£¿£¿TIM2ÎªÖ÷¶¨Ê±Æ÷£¬TIM3Îª´Ó¶¨Ê±Æ÷
	DAC_SetChannel1Data(DAC_Align_12b_R,carSpeed);       //³õÊ¼ÖµÎª0
	
/********Ö÷Ñ­»·********/

	while(1)
	{

		if(Rangel>Langel)
		    angel_re=Rangel-Langel+5000;
		else 
			angel_re=Langel-Rangel+5000;
		
		if(USART_RX_STA&0x8000)               //½ÓÊÕÍê³É   
		{	
            temp160=0;
	        temp161=0;
	        temp162=0;			
		  	len=USART_RX_STA&0x3fff;          //µÃµ½´Ë´Î½ÓÊÕµ½µÄÊı¾İ³¤¶È
			printf("\r\nÄú·¢ËÍµÄÏûÏ¢Îª:\r\n");        

	      /********Êı¾İ´¦Àí********/	
			
			for(t=0;t<len;t++)
			{
				res=USART_RX_BUF[t];          //Ïò´®¿Ú1·¢ËÍÊı¾İ£¬¼´´®¿Ú1½ÓÊÜ´Ó´®¿ÚÖúÊÖ·¢³öµÄÃüÁî
                if(t==0)
				{
					temp160=res;		         
				}
                else if(t==1) 
				{
					temp=res<<8;
					temp160=temp160|temp;     //byte1¡¢2				
				}
				else if(t==2)
				{
					temp161=res;
				}
				else if(t==3)
				{
					temp=res<<8;
					temp161=temp161|temp;     //byte3¡¢4
				}
				else if(t==4)
				{
                    temp162=res;
				}
                else if(t==5)
				{
					temp=res<<8; 
					temp162=temp162|temp;     //byte5¡¢6
				}					
				while(USART_GetFlagStatus(USART1,USART_FLAG_TC)!=SET);  //µÈ´ı·¢ËÍ½áÊø			
			}
			
	
			/********Ö¸ÁîÅĞ¶Ï********/	
			
			
			if((temp162&0x0040)==0x0040)                        //×ªÏò¸´Î»
			 {
					   printf("×ªÏò¸´Î»");
					   if((temp162&0x0020)==0x0020)                      //½Ç¶È·ûºÅÎ»
				   { 
					   printf("×ª½Ç·ûºÅÎ»");
					   if(PA5==1)           //Èô²»ÊÇÊ¹ÄÜ£¬½«¸ÃÎ» ÖÃ0Ê¹ÄÜ
						{
							GPIO_ResetBits(GPIOA,GPIO_Pin_5);
							PA6=!PA6;
							Output_Pulse(angel_re);
						}else 
						{
							PA6=!PA6;
							Output_Pulse(angel_re);
						}
				   }
						 else
				   {
							if(PA5==1)
							{
								PA6=!PA6;
								GPIO_ResetBits(GPIOA,GPIO_Pin_5);
								Output_Pulse(angel_re);
							}
							else 
							{
								PA6=!PA6;
								Output_Pulse(angel_re);	
							}
				   }
				  for(i=0;i<20000;++i)
				   {
						 printf("\r\nÎ´¸´Î»\r\n");
						  if(PE3==0)   //Ó¦¸ÃÊÇ×ªÏò¸´Î»µÄÒı½Å
						{
							TIM_Cmd(TIM2, DISABLE);
							TIM_Cmd(TIM3, DISABLE);
							angel_re=0;
									printf("\r\nÒÑ¸´Î»\r\n");
									break;
						}
				   
				  }
			}
			 
		    if((temp162&0x0080)==0x0080)	                    //É²³µ
			 {
					 
				 statue1 = GPIO_ReadOutputDataBit(GPIOF,  GPIO_Pin_5);
				 if(statue1!=1){					  		
					   GPIO_WriteBit( GPIOF,GPIO_Pin_5,  Bit_SET);  //Èç¹ûµ±Ç°²»ÊÇÉ²³µ £¬ÔòÉ²³µÎ» ÖÃ1 £¬É²³µ
					   statue1=1;
				}    
					  printf("\r\nÉ²³µ\r\n");
				 statue1 = GPIO_ReadOutputDataBit(GPIOF,  GPIO_Pin_5);
					  printf("É²³µµÄµ±Ç°×´Ì¬ÊÇ%d",statue1); 
			 }
			 else {	
                     statue1 = GPIO_ReadOutputDataBit(GPIOF,  GPIO_Pin_5);				 //²»É²³µ
			  		 if(statue1==1) { 
					 GPIO_WriteBit( GPIOF,GPIO_Pin_5,  Bit_RESET);  //Èç¹ûµ±Ç°ÊÇÉ²³µ£¬ÔòÉ²³µÎ»ÖÃ0             
				 }
			 }
            if((temp162&0x0100)==0x0100)
				{ 			                                    //×Ô¼ì
				 selfcheck(); 
				}
			if((temp162&0x0200)==0x0200)                       //µÍËÙ        £¿£¿£¿£¿£¿
			 {
					DISU=!DISU;
				
				 printf("µÍËÙµÄµ±Ç°×´Ì¬ÊÇ%d",statue2);
					printf("\r\nµÍËÙµ²\r\n"); 
				
			 }	
			if((temp162&0x0400)==0x0400)                       //¸ßËÙ        £¿£¿£¿£¿£¿£¿£¿
			 {
					GAOSU=!GAOSU;
				 
				 
					printf("\r\n¸ßËÙµ²\r\n");
			 }   			
			if((temp162&0x0800)==0x0800)                       //¶ÏÊ¹ÄÜ       £¿£¿£¿£¿£¿£¿£¿£¿£¿
			 {
				    statue2 = GPIO_ReadOutputDataBit(GPIOA,  GPIO_Pin_5);
				    printf(" Ê¹ÄÜµÄµ±Ç°×´Ì¬ÊÇ%d",statue2); 
					GPIO_SetBits(GPIOA,GPIO_Pin_5);		       //ÖÃÎ»1
					printf("\r\nÊ¹ÄÜÒÑ¶Ï\r\n");
				     printf(" Ê¹ÄÜµÄµ±Ç°×´Ì¬ÊÇ%d",statue2); 
			 }            				             
            if((temp161&0x2000)==0x2000)                        //Ç°½ø/ºóÍË
			 {	
				 statue5 = GPIO_ReadOutputDataBit(GPIOF,  GPIO_Pin_7);
				 printf("µ¹³µµÄµ±Ç°×´Ì¬ÊÇ%d",statue5);
				 
						DAOCHE=!DAOCHE;		
					
					delay_ms(500);
					printf("\r\nµ¹³µ\r\n");				
			 }	
			if((temp161&0x1000)==0x1000)	                    //Æô¶¯²¢ÉèÖÃËÙ¶ÈÖµ
			 {
				    printf("\r\nÆô¶¯\r\n");
				    carSpeed=Hex_Dec(temp161&0x0fff);                 
			        DAC_SetChannel1Data(DAC_Align_12b_R, carSpeed);		//ÉèÖÃËÙ¶ÈÖµ
		            printf("\r\nËÙ¶ÈÖµÎª£º%d\r\n",carSpeed);				
			 }                  					
			if((temp161&0x4000)==0x4000)	                    //ÃùµÑ
			 {
				    
				 statue6 = GPIO_ReadOutputDataBit(GPIOF,  GPIO_Pin_4);
				 printf("ÃùµÑµÄµ±Ç°×´Ì¬ÊÇ%d",statue6);
				 if(statue6==0)    {}     //Èç¹ûµ±Ç°ÊÇÃùµÑ£¬Ôò²»²Ù×÷
					 else {
						BEEP=!BEEP;		//Èç¹ûµ±Ç°²»ÊÇÃùµÑ£¬ÔòÃùµÑÎ»È¡·´
					 }
                    printf("\r\nÃùµÑ\r\n");				
			 }
			if((temp161&0x8000)==0x8000)                        //Ç°´óµÆ
			 {
				 
				 statue7 = GPIO_ReadOutputDataBit(GPIOF,  GPIO_Pin_1);
				
						 LED1=!LED1;		
				    printf("\r\nÇ°´óµÆ\r\n");				
			 }								
    	    if((temp160&0x1000)==0x1000)	                    //ÓÒÉÁ
			 {
		            LED2=!LED2; 
				 statue8 = GPIO_ReadOutputDataBit(GPIOF,  GPIO_Pin_2);
				 printf("ÓÒÉÁµÄµ±Ç°×´Ì¬ÊÇ%d",statue8);
				 
								
		
			        printf("\r\nÓÒÉÁµÆ\r\n");				
			 }
			if((temp160&0x2000)==0x2000)	                    //×óÉÁ
			{
		            LED3=!LED3;
				statue9 = GPIO_ReadOutputDataBit(GPIOF,  GPIO_Pin_3);
				 printf("×óÉÁµÄµ±Ç°×´Ì¬statueÊÇ%d",statue9);
				  if(statue9==0)    {}     //Èç¹ûµ±Ç°ÊÇÇ°´óµÆÁÁ£¬Ôò²»²Ù×÷
					 else if(statue8==1){
						 LED2=!LED2;		//Èç¹ûµ±Ç°Ç°´óµÆ²»ÁÁ£¬ÔòÇ°´óµÆÎ»È¡·´
					 }
			        printf("\r\n×óÉÁµÆ\r\n");				
			}			
            if((temp160&0x4000)==0x4000)	                    //ÓÒ×ª
			{
                    printf("\r\nÓÒ×ª\r\n");
					GPIO_ResetBits(GPIOA,GPIO_Pin_6);		
				if(PE3==0)
					Rangel=0;					
				if(carSpeed>=2000)			
				{
					DAC_SetChannel1Data(DAC_Align_12b_R, 1800);
					printf("\r\nµ±Ç°ËÙ¶ÈÖµ£º%d\r\n",1800);
				}								    
                else
                    DAC_SetChannel1Data(DAC_Align_12b_R, 1600);							
				
				angel_hex=temp160&0x001f;                       //½Ç¶È
				angel=Hex_Dec(angel_hex);
				printf("\r\n×ªÏò½Ç¶ÈÖµ£º%d¡ã\r\n",angel);
				
				rot_speed_hext=temp162&0x0fff;	            		//×ªËÙ		
				rot_speed=Hex_Dec(rot_speed_hext);              	
				Frequence_Setting(rot_speed);		
                rot_speed=(1e6)/rot_speed;				
				printf("\r\nÆµÂÊ£º%dHz\r\n",rot_speed);
							
				angel=angel/0.0225;                            //Âö³åÊı
				Output_Pulse(angel);  
				Rangel=angel+Rangel;
			}
            if((temp160&0x8000)==0x8000)	                    //×ó×ª
			{

                   printf("\r\n×ó×ª\r\n");	
				   GPIO_SetBits(GPIOA,GPIO_Pin_6);	
				if(PE3==0)
					Langel=0;					
				if(carSpeed>=2000)			
				{
					DAC_SetChannel1Data(DAC_Align_12b_R, 1800);
					printf("\r\nµ±Ç°ËÙ¶ÈÖµ£º%d\r\n",1800);
				}								    
				else
				    DAC_SetChannel1Data(DAC_Align_12b_R, 1600);	
						
						angel_hex=temp160&0x001f;                       //½Ç¶È
						angel=Hex_Dec(angel_hex);
						printf("\r\n×ªÏò½Ç¶ÈÖµ£º%d¡ã\r\n",angel);
						
						rot_speed_hext=temp162&0x0fff;	            		//×ªËÙ		
						rot_speed=Hex_Dec(rot_speed_hext);              	
						Frequence_Setting(rot_speed);		
				        rot_speed=(1e6)/rot_speed;				
						printf("\r\nÆµÂÊ£º%dHz\r\n",rot_speed);
						angel=angel/0.0225;                            //Âö³åÊı

						Output_Pulse(angel);
						Langel=angel+Langel;
			}			
					printf("\r\n\r\n");//²åÈë»»ĞĞ
					USART_RX_STA=0;
	    }    //½ÓÊÕÊı¾İ if£¨£©½áÊø
	}     // while()½áÊø
}     //main£¨£©½áÊø

