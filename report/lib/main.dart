import 'package:report/report_analyzer_page.dart';
import 'package:flutter/material.dart';

void main() async {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      initialRoute: '/',    
      routes: { //splash onboarding1-2-3 select_profesion login signin consent userData 
        // '/':(context) =>  ChatScreen(),
        '/':(context) =>  StreamlitWebView(),
        // '/':(context) => const SplashPage(),
        
      }
    );
  }
}
