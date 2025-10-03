import 'package:flutter/material.dart';
import 'package:flutter_inappwebview/flutter_inappwebview.dart';

class StreamlitWebView extends StatefulWidget {
  const StreamlitWebView({Key? key}) : super(key: key);

  @override
  State<StreamlitWebView> createState() => _StreamlitWebViewState();
}

class _StreamlitWebViewState extends State<StreamlitWebView> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('ChronoCancer Analyzer')),
      body: InAppWebView(
        initialUrlRequest: URLRequest(
          url: WebUri('https://chronocancerbackend-production.up.railway.app'), // ✅ Use WebUri
        ),
        initialOptions: InAppWebViewGroupOptions(
          crossPlatform: InAppWebViewOptions(
            javaScriptEnabled: true,
          ),
        ),
      ),
    );
  }
}
