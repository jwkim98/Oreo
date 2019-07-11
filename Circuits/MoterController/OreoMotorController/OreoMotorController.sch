EESchema Schematic File Version 4
LIBS:OreoMotorController-cache
EELAYER 29 0
EELAYER END
$Descr A4 11693 8268
encoding utf-8
Sheet 1 1
Title ""
Date ""
Rev ""
Comp ""
Comment1 ""
Comment2 ""
Comment3 ""
Comment4 ""
$EndDescr
$Comp
L MCU_Module:Arduino_Nano_v3.x A?
U 1 1 5D123426
P 4800 3350
F 0 "A?" H 4800 2261 50  0000 C CNN
F 1 "Arduino_Nano_v3.x" H 4800 2170 50  0000 C CNN
F 2 "Module:Arduino_Nano" H 4950 2400 50  0001 L CNN
F 3 "http://www.mouser.com/pdfdocs/Gravitech_Arduino_Nano3_0.pdf" H 4800 2350 50  0001 C CNN
	1    4800 3350
	1    0    0    -1  
$EndComp
$Comp
L Regulator_Switching:LM2576HVS-5 U?
U 1 1 5D126151
P 5300 1450
F 0 "U?" H 5300 1817 50  0000 C CNN
F 1 "LM2576HVS-5" H 5300 1726 50  0000 C CNN
F 2 "Package_TO_SOT_SMD:TO-263-5_TabPin3" H 5300 1200 50  0001 L CIN
F 3 "http://www.ti.com/lit/ds/symlink/lm2576.pdf" H 5300 1450 50  0001 C CNN
	1    5300 1450
	1    0    0    -1  
$EndComp
Wire Wire Line
	4300 2950 3700 2950
Wire Wire Line
	4300 3050 3700 3050
Wire Wire Line
	4300 3150 3700 3150
Wire Wire Line
	4300 3250 3700 3250
$Comp
L Device:CP C?
U 1 1 5D136A4B
P 4200 1750
F 0 "C?" H 4318 1796 50  0000 L CNN
F 1 "CP" H 4318 1705 50  0000 L CNN
F 2 "" H 4238 1600 50  0001 C CNN
F 3 "~" H 4200 1750 50  0001 C CNN
	1    4200 1750
	1    0    0    -1  
$EndComp
Wire Wire Line
	4200 1900 5300 1900
Wire Wire Line
	5300 1750 5300 1900
Connection ~ 5300 1900
Wire Wire Line
	5300 1900 6400 1900
$Comp
L power:GND #PWR0101
U 1 1 5D137893
P 6450 1900
F 0 "#PWR0101" H 6450 1650 50  0001 C CNN
F 1 "GND" V 6455 1772 50  0000 R CNN
F 2 "" H 6450 1900 50  0001 C CNN
F 3 "" H 6450 1900 50  0001 C CNN
	1    6450 1900
	0    -1   -1   0   
$EndComp
$Comp
L Connector:Conn_01x02_Female J?
U 1 1 5D138D17
P 3100 1500
F 0 "J?" H 2992 1175 50  0000 C CNN
F 1 "Conn_01x02_Female" H 2992 1266 50  0000 C CNN
F 2 "" H 3100 1500 50  0001 C CNN
F 3 "~" H 3100 1500 50  0001 C CNN
	1    3100 1500
	-1   0    0    1   
$EndComp
Wire Wire Line
	3300 1400 3800 1400
Wire Wire Line
	4800 1400 4800 1350
Wire Wire Line
	3300 1500 4200 1500
Wire Wire Line
	4200 1500 4200 1600
Wire Wire Line
	4200 1500 4800 1500
Wire Wire Line
	4800 1500 4800 1550
Connection ~ 4200 1500
Wire Wire Line
	2900 2950 2200 2950
Wire Wire Line
	2900 3050 2200 3050
Wire Wire Line
	2900 3150 2200 3150
Wire Wire Line
	2900 3250 2200 3250
$Comp
L MCU_Module:Arduino_Nano_v3.x A?
U 1 1 5D140C2A
P 6800 3350
F 0 "A?" H 6800 2261 50  0000 C CNN
F 1 "Arduino_Nano_v3.x" H 6800 2170 50  0000 C CNN
F 2 "Module:Arduino_Nano" H 6950 2400 50  0001 L CNN
F 3 "http://www.mouser.com/pdfdocs/Gravitech_Arduino_Nano3_0.pdf" H 6800 2350 50  0001 C CNN
	1    6800 3350
	1    0    0    -1  
$EndComp
$Comp
L Connector:Conn_01x03_Male J?
U 1 1 5D1446B5
P 5450 5050
F 0 "J?" H 5558 5331 50  0000 C CNN
F 1 "Conn_01x03_Male" H 5558 5240 50  0000 C CNN
F 2 "" H 5450 5050 50  0001 C CNN
F 3 "~" H 5450 5050 50  0001 C CNN
	1    5450 5050
	0    -1   -1   0   
$EndComp
$Comp
L Connector:Conn_01x06_Female J?
U 1 1 5D149033
P 9200 3150
F 0 "J?" H 9228 3126 50  0000 L CNN
F 1 "Conn_01x06_Female" H 9228 3035 50  0000 L CNN
F 2 "" H 9200 3150 50  0001 C CNN
F 3 "~" H 9200 3150 50  0001 C CNN
	1    9200 3150
	-1   0    0    1   
$EndComp
Wire Wire Line
	9400 3350 9950 3350
Wire Wire Line
	9950 3350 9950 3400
Wire Wire Line
	9400 3250 9950 3250
Wire Wire Line
	9400 3150 9950 3150
Wire Wire Line
	9400 3050 9950 3050
Wire Wire Line
	9400 2850 9950 2850
Text Label 9600 3350 0    50   ~ 0
VCC
Text Label 9600 3250 0    50   ~ 0
GND
Text Label 9600 3150 0    50   ~ 0
SCL
Text Label 9600 3050 0    50   ~ 0
SDA
Text Label 9600 2950 0    50   ~ 0
GPIO
Text Label 9600 2850 0    50   ~ 0
XHSUT
$Comp
L power:+5V #PWR0102
U 1 1 5D152B9E
P 9950 3400
F 0 "#PWR0102" H 9950 3250 50  0001 C CNN
F 1 "+5V" V 9965 3528 50  0000 L CNN
F 2 "" H 9950 3400 50  0001 C CNN
F 3 "" H 9950 3400 50  0001 C CNN
	1    9950 3400
	0    1    1    0   
$EndComp
Wire Wire Line
	7300 3750 7750 3750
Wire Wire Line
	7300 3850 7750 3850
Text Label 7450 3750 0    50   ~ 0
SDA
Text Label 7450 3850 0    50   ~ 0
SCL
$Comp
L Device:Q_NPN_EBC Q?
U 1 1 5D165B6E
P 10350 2950
F 0 "Q?" H 10541 2996 50  0000 L CNN
F 1 "Q_NPN_EBC" H 10541 2905 50  0000 L CNN
F 2 "" H 10550 3050 50  0001 C CNN
F 3 "~" H 10350 2950 50  0001 C CNN
	1    10350 2950
	1    0    0    -1  
$EndComp
Wire Wire Line
	9400 2950 10150 2950
$Comp
L Device:R R?
U 1 1 5D16997F
P 10450 2500
F 0 "R?" H 10520 2546 50  0000 L CNN
F 1 "R" H 10520 2455 50  0000 L CNN
F 2 "" V 10380 2500 50  0001 C CNN
F 3 "~" H 10450 2500 50  0001 C CNN
	1    10450 2500
	1    0    0    -1  
$EndComp
Wire Wire Line
	10450 2650 10450 2700
$Comp
L power:GND #PWR0103
U 1 1 5D16A29B
P 10600 3250
F 0 "#PWR0103" H 10600 3000 50  0001 C CNN
F 1 "GND" V 10605 3122 50  0000 R CNN
F 2 "" H 10600 3250 50  0001 C CNN
F 3 "" H 10600 3250 50  0001 C CNN
	1    10600 3250
	0    -1   -1   0   
$EndComp
Wire Wire Line
	10450 3150 10450 3250
Wire Wire Line
	10450 3250 10600 3250
$Comp
L power:+5V #PWR0104
U 1 1 5D16ACA0
P 10050 2350
F 0 "#PWR0104" H 10050 2200 50  0001 C CNN
F 1 "+5V" V 10065 2478 50  0000 L CNN
F 2 "" H 10050 2350 50  0001 C CNN
F 3 "" H 10050 2350 50  0001 C CNN
	1    10050 2350
	0    -1   -1   0   
$EndComp
Wire Wire Line
	10050 2350 10450 2350
Wire Wire Line
	10450 2700 10000 2700
Connection ~ 10450 2700
Wire Wire Line
	10450 2700 10450 2750
Text Label 10050 2700 0    50   ~ 0
ISACTIVE
Wire Wire Line
	6300 2950 5800 2950
Text Label 5900 2950 0    50   ~ 0
ISACTIVE
Wire Wire Line
	5800 3050 6300 3050
Text Label 5900 3050 0    50   ~ 0
ISACTIVE2
Wire Wire Line
	6300 3150 5800 3150
Wire Wire Line
	6300 3250 5800 3250
$Comp
L Connector:Conn_01x06_Female J?
U 1 1 5D178EAD
P 9200 1550
F 0 "J?" H 9228 1526 50  0000 L CNN
F 1 "Conn_01x06_Female" H 9228 1435 50  0000 L CNN
F 2 "" H 9200 1550 50  0001 C CNN
F 3 "~" H 9200 1550 50  0001 C CNN
	1    9200 1550
	-1   0    0    1   
$EndComp
Wire Wire Line
	9400 1750 9950 1750
Wire Wire Line
	9950 1750 9950 1800
Wire Wire Line
	9400 1650 9950 1650
Wire Wire Line
	9400 1550 9950 1550
Wire Wire Line
	9400 1450 9950 1450
Wire Wire Line
	9400 1250 9950 1250
Text Label 9600 1750 0    50   ~ 0
VCC
Text Label 9600 1650 0    50   ~ 0
GND
Text Label 9600 1550 0    50   ~ 0
SCL
Text Label 9600 1450 0    50   ~ 0
SDA
Text Label 9600 1250 0    50   ~ 0
XHSUT1
$Comp
L power:+5V #PWR0105
U 1 1 5D178EC3
P 9950 1800
F 0 "#PWR0105" H 9950 1650 50  0001 C CNN
F 1 "+5V" V 9965 1928 50  0000 L CNN
F 2 "" H 9950 1800 50  0001 C CNN
F 3 "" H 9950 1800 50  0001 C CNN
	1    9950 1800
	0    1    1    0   
$EndComp
$Comp
L Device:Q_NPN_EBC Q?
U 1 1 5D178ECD
P 10350 1350
F 0 "Q?" H 10541 1396 50  0000 L CNN
F 1 "Q_NPN_EBC" H 10541 1305 50  0000 L CNN
F 2 "" H 10550 1450 50  0001 C CNN
F 3 "~" H 10350 1350 50  0001 C CNN
	1    10350 1350
	1    0    0    -1  
$EndComp
Wire Wire Line
	9400 1350 10150 1350
$Comp
L Device:R R?
U 1 1 5D178ED8
P 10450 900
F 0 "R?" H 10520 946 50  0000 L CNN
F 1 "R" H 10520 855 50  0000 L CNN
F 2 "" V 10380 900 50  0001 C CNN
F 3 "~" H 10450 900 50  0001 C CNN
	1    10450 900 
	1    0    0    -1  
$EndComp
Wire Wire Line
	10450 1050 10450 1100
$Comp
L power:GND #PWR0106
U 1 1 5D178EE3
P 10600 1650
F 0 "#PWR0106" H 10600 1400 50  0001 C CNN
F 1 "GND" V 10605 1522 50  0000 R CNN
F 2 "" H 10600 1650 50  0001 C CNN
F 3 "" H 10600 1650 50  0001 C CNN
	1    10600 1650
	0    -1   -1   0   
$EndComp
Wire Wire Line
	10450 1550 10450 1650
Wire Wire Line
	10450 1650 10600 1650
$Comp
L power:+5V #PWR0107
U 1 1 5D178EEF
P 10050 750
F 0 "#PWR0107" H 10050 600 50  0001 C CNN
F 1 "+5V" V 10065 878 50  0000 L CNN
F 2 "" H 10050 750 50  0001 C CNN
F 3 "" H 10050 750 50  0001 C CNN
	1    10050 750 
	0    -1   -1   0   
$EndComp
Wire Wire Line
	10050 750  10450 750 
Wire Wire Line
	10450 1100 10000 1100
Connection ~ 10450 1100
Wire Wire Line
	10450 1100 10450 1150
Text Label 10050 1100 0    50   ~ 0
ISACTIVE1
$Comp
L Connector:Conn_01x06_Female J?
U 1 1 5D17F273
P 9300 4550
F 0 "J?" H 9328 4526 50  0000 L CNN
F 1 "Conn_01x06_Female" H 9328 4435 50  0000 L CNN
F 2 "" H 9300 4550 50  0001 C CNN
F 3 "~" H 9300 4550 50  0001 C CNN
	1    9300 4550
	-1   0    0    1   
$EndComp
Wire Wire Line
	9500 4750 10050 4750
Wire Wire Line
	10050 4750 10050 4800
Wire Wire Line
	9500 4650 10050 4650
Wire Wire Line
	9500 4550 10050 4550
Wire Wire Line
	9500 4450 10050 4450
Wire Wire Line
	9500 4250 10050 4250
Text Label 9700 4750 0    50   ~ 0
VCC
Text Label 9700 4650 0    50   ~ 0
GND
Text Label 9700 4550 0    50   ~ 0
SCL
Text Label 9700 4450 0    50   ~ 0
SDA
Text Label 9700 4350 0    50   ~ 0
GPIO
Text Label 9700 4250 0    50   ~ 0
XHSUT
$Comp
L power:+5V #PWR0108
U 1 1 5D17F289
P 10050 4800
F 0 "#PWR0108" H 10050 4650 50  0001 C CNN
F 1 "+5V" V 10065 4928 50  0000 L CNN
F 2 "" H 10050 4800 50  0001 C CNN
F 3 "" H 10050 4800 50  0001 C CNN
	1    10050 4800
	0    1    1    0   
$EndComp
$Comp
L Device:Q_NPN_EBC Q?
U 1 1 5D17F293
P 10450 4350
F 0 "Q?" H 10641 4396 50  0000 L CNN
F 1 "Q_NPN_EBC" H 10641 4305 50  0000 L CNN
F 2 "" H 10650 4450 50  0001 C CNN
F 3 "~" H 10450 4350 50  0001 C CNN
	1    10450 4350
	1    0    0    -1  
$EndComp
Wire Wire Line
	9500 4350 10250 4350
$Comp
L Device:R R?
U 1 1 5D17F29E
P 10550 3900
F 0 "R?" H 10620 3946 50  0000 L CNN
F 1 "R" H 10620 3855 50  0000 L CNN
F 2 "" V 10480 3900 50  0001 C CNN
F 3 "~" H 10550 3900 50  0001 C CNN
	1    10550 3900
	1    0    0    -1  
$EndComp
Wire Wire Line
	10550 4050 10550 4100
$Comp
L power:GND #PWR0109
U 1 1 5D17F2A9
P 10700 4650
F 0 "#PWR0109" H 10700 4400 50  0001 C CNN
F 1 "GND" V 10705 4522 50  0000 R CNN
F 2 "" H 10700 4650 50  0001 C CNN
F 3 "" H 10700 4650 50  0001 C CNN
	1    10700 4650
	0    -1   -1   0   
$EndComp
Wire Wire Line
	10550 4550 10550 4650
Wire Wire Line
	10550 4650 10700 4650
$Comp
L power:+5V #PWR0110
U 1 1 5D17F2B5
P 10150 3750
F 0 "#PWR0110" H 10150 3600 50  0001 C CNN
F 1 "+5V" V 10165 3878 50  0000 L CNN
F 2 "" H 10150 3750 50  0001 C CNN
F 3 "" H 10150 3750 50  0001 C CNN
	1    10150 3750
	0    -1   -1   0   
$EndComp
Wire Wire Line
	10150 3750 10550 3750
Wire Wire Line
	10550 4100 10100 4100
Connection ~ 10550 4100
Wire Wire Line
	10550 4100 10550 4150
Text Label 10150 4100 0    50   ~ 0
ISACTIVE
$Comp
L Connector:Conn_01x06_Female J?
U 1 1 5D1871BD
P 9350 6100
F 0 "J?" H 9378 6076 50  0000 L CNN
F 1 "Conn_01x06_Female" H 9378 5985 50  0000 L CNN
F 2 "" H 9350 6100 50  0001 C CNN
F 3 "~" H 9350 6100 50  0001 C CNN
	1    9350 6100
	-1   0    0    1   
$EndComp
Wire Wire Line
	9550 6300 10100 6300
Wire Wire Line
	10100 6300 10100 6350
Wire Wire Line
	9550 6200 10100 6200
Wire Wire Line
	9550 6100 10100 6100
Wire Wire Line
	9550 6000 10100 6000
Wire Wire Line
	9550 5800 10100 5800
Text Label 9750 6300 0    50   ~ 0
VCC
Text Label 9750 6200 0    50   ~ 0
GND
Text Label 9750 6100 0    50   ~ 0
SCL
Text Label 9750 6000 0    50   ~ 0
SDA
Text Label 9750 5900 0    50   ~ 0
GPIO
Text Label 9750 5800 0    50   ~ 0
XHSUT
$Comp
L power:+5V #PWR0111
U 1 1 5D1871D3
P 10100 6350
F 0 "#PWR0111" H 10100 6200 50  0001 C CNN
F 1 "+5V" V 10115 6478 50  0000 L CNN
F 2 "" H 10100 6350 50  0001 C CNN
F 3 "" H 10100 6350 50  0001 C CNN
	1    10100 6350
	0    1    1    0   
$EndComp
$Comp
L Device:Q_NPN_EBC Q?
U 1 1 5D1871DD
P 10500 5900
F 0 "Q?" H 10691 5946 50  0000 L CNN
F 1 "Q_NPN_EBC" H 10691 5855 50  0000 L CNN
F 2 "" H 10700 6000 50  0001 C CNN
F 3 "~" H 10500 5900 50  0001 C CNN
	1    10500 5900
	1    0    0    -1  
$EndComp
Wire Wire Line
	9550 5900 10300 5900
$Comp
L Device:R R?
U 1 1 5D1871E8
P 10600 5450
F 0 "R?" H 10670 5496 50  0000 L CNN
F 1 "R" H 10670 5405 50  0000 L CNN
F 2 "" V 10530 5450 50  0001 C CNN
F 3 "~" H 10600 5450 50  0001 C CNN
	1    10600 5450
	1    0    0    -1  
$EndComp
Wire Wire Line
	10600 5600 10600 5650
$Comp
L power:GND #PWR0112
U 1 1 5D1871F3
P 10750 6200
F 0 "#PWR0112" H 10750 5950 50  0001 C CNN
F 1 "GND" V 10755 6072 50  0000 R CNN
F 2 "" H 10750 6200 50  0001 C CNN
F 3 "" H 10750 6200 50  0001 C CNN
	1    10750 6200
	0    -1   -1   0   
$EndComp
Wire Wire Line
	10600 6100 10600 6200
Wire Wire Line
	10600 6200 10750 6200
$Comp
L power:+5V #PWR0113
U 1 1 5D1871FF
P 10200 5300
F 0 "#PWR0113" H 10200 5150 50  0001 C CNN
F 1 "+5V" V 10215 5428 50  0000 L CNN
F 2 "" H 10200 5300 50  0001 C CNN
F 3 "" H 10200 5300 50  0001 C CNN
	1    10200 5300
	0    -1   -1   0   
$EndComp
Wire Wire Line
	10200 5300 10600 5300
Wire Wire Line
	10600 5650 10150 5650
Connection ~ 10600 5650
Wire Wire Line
	10600 5650 10600 5700
Text Label 10200 5650 0    50   ~ 0
ISACTIVE
$Comp
L Transistor_Array:ULN2003 U?
U 1 1 5D276BEB
P 3300 3050
F 0 "U?" H 3300 3717 50  0000 C CNN
F 1 "ULN2003" H 3300 3626 50  0000 C CNN
F 2 "" H 3350 2500 50  0001 L CNN
F 3 "http://www.ti.com/lit/ds/symlink/uln2003a.pdf" H 3400 2850 50  0001 C CNN
	1    3300 3050
	-1   0    0    1   
$EndComp
$Comp
L power:GND #PWR0114
U 1 1 5D2895CE
P 3300 2200
F 0 "#PWR0114" H 3300 1950 50  0001 C CNN
F 1 "GND" H 3305 2027 50  0000 C CNN
F 2 "" H 3300 2200 50  0001 C CNN
F 3 "" H 3300 2200 50  0001 C CNN
	1    3300 2200
	-1   0    0    1   
$EndComp
Wire Wire Line
	3300 2200 3300 2450
$Comp
L Device:L L?
U 1 1 5D28C232
P 6150 1550
F 0 "L?" V 6340 1550 50  0000 C CNN
F 1 "L" V 6249 1550 50  0000 C CNN
F 2 "" H 6150 1550 50  0001 C CNN
F 3 "~" H 6150 1550 50  0001 C CNN
	1    6150 1550
	0    -1   -1   0   
$EndComp
$Comp
L Device:CP C?
U 1 1 5D28CB35
P 6400 1700
F 0 "C?" H 6518 1746 50  0000 L CNN
F 1 "CP" H 6518 1655 50  0000 L CNN
F 2 "" H 6438 1550 50  0001 C CNN
F 3 "~" H 6400 1700 50  0001 C CNN
	1    6400 1700
	1    0    0    -1  
$EndComp
Wire Wire Line
	5800 1550 6000 1550
Wire Wire Line
	6300 1550 6400 1550
Wire Wire Line
	6400 1850 6400 1900
Connection ~ 6400 1900
Wire Wire Line
	6400 1900 6450 1900
Wire Wire Line
	5800 1350 6400 1350
Wire Wire Line
	6400 1350 6400 1550
Connection ~ 6400 1550
Wire Wire Line
	6400 1550 7050 1550
$Comp
L power:+5V #PWR0115
U 1 1 5D297BD8
P 7050 1550
F 0 "#PWR0115" H 7050 1400 50  0001 C CNN
F 1 "+5V" V 7065 1678 50  0000 L CNN
F 2 "" H 7050 1550 50  0001 C CNN
F 3 "" H 7050 1550 50  0001 C CNN
	1    7050 1550
	0    1    1    0   
$EndComp
Wire Wire Line
	5300 3350 5550 3350
Wire Wire Line
	5550 3350 5550 4850
$Comp
L power:+5V #PWR0116
U 1 1 5D2A2702
P 5350 4550
F 0 "#PWR0116" H 5350 4400 50  0001 C CNN
F 1 "+5V" H 5365 4723 50  0000 C CNN
F 2 "" H 5350 4550 50  0001 C CNN
F 3 "" H 5350 4550 50  0001 C CNN
	1    5350 4550
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR0117
U 1 1 5D2A2E3D
P 5450 4600
F 0 "#PWR0117" H 5450 4350 50  0001 C CNN
F 1 "GND" H 5455 4427 50  0000 C CNN
F 2 "" H 5450 4600 50  0001 C CNN
F 3 "" H 5450 4600 50  0001 C CNN
	1    5450 4600
	-1   0    0    1   
$EndComp
Wire Wire Line
	5350 4550 5350 4850
Wire Wire Line
	5450 4600 5450 4850
$Comp
L power:+5V #PWR0118
U 1 1 5D2AC0D9
P 5000 2200
F 0 "#PWR0118" H 5000 2050 50  0001 C CNN
F 1 "+5V" H 5015 2373 50  0000 C CNN
F 2 "" H 5000 2200 50  0001 C CNN
F 3 "" H 5000 2200 50  0001 C CNN
	1    5000 2200
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR0119
U 1 1 5D2AC8D9
P 4900 4650
F 0 "#PWR0119" H 4900 4400 50  0001 C CNN
F 1 "GND" H 4905 4477 50  0000 C CNN
F 2 "" H 4900 4650 50  0001 C CNN
F 3 "" H 4900 4650 50  0001 C CNN
	1    4900 4650
	1    0    0    -1  
$EndComp
Wire Wire Line
	4800 4350 4800 4450
Wire Wire Line
	4800 4450 4900 4450
Wire Wire Line
	4900 4450 4900 4650
Wire Wire Line
	4900 4350 4900 4450
Connection ~ 4900 4450
Wire Wire Line
	5000 2350 5000 2200
$Comp
L Connector:Conn_01x04_Female J?
U 1 1 5D2B7AFA
P 2000 3150
F 0 "J?" H 1892 2725 50  0000 C CNN
F 1 "Conn_01x04_Female" H 1892 2816 50  0000 C CNN
F 2 "" H 2000 3150 50  0001 C CNN
F 3 "~" H 2000 3150 50  0001 C CNN
	1    2000 3150
	-1   0    0    1   
$EndComp
$Comp
L power:VDD #PWR0120
U 1 1 5D2B8A4F
P 3800 1250
F 0 "#PWR0120" H 3800 1100 50  0001 C CNN
F 1 "VDD" H 3817 1423 50  0000 C CNN
F 2 "" H 3800 1250 50  0001 C CNN
F 3 "" H 3800 1250 50  0001 C CNN
	1    3800 1250
	1    0    0    -1  
$EndComp
Wire Wire Line
	3800 1250 3800 1400
Connection ~ 3800 1400
Wire Wire Line
	3800 1400 4800 1400
$EndSCHEMATC
