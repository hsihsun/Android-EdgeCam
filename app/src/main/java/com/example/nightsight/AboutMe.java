package com.example.nightsight;

import android.app.Activity;
import android.os.Bundle;
import android.text.Html;
import android.widget.TextView;

public class AboutMe extends Activity{

        private TextView texture;

        @Override
        public void onCreate(Bundle savedInstanceState) {
            super.onCreate(savedInstanceState);
            setContentView(R.layout.activity_about_me);
            texture = (TextView) findViewById(R.id.textView2);

            texture.setText(Html.fromHtml("Hi, here is <font color=\"red\">EdgeCam </font> App!"));
            texture.setTextSize(20.0f);
        }
}
