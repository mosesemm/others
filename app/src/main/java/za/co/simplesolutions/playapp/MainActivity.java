package za.co.simplesolutions.playapp;

import android.content.Intent;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

public class MainActivity extends AppCompatActivity {

    private static final String LOG_TAG = MainActivity.class.getSimpleName();
    public static final String EXTRA_MESSAGE = "za.co.simplesolutions.extra.MESSAGE";
    private EditText messageEditText;
    private static final int TEXT_REQUEST = 1;
    private TextView replyHeaderTxt;
    private TextView replyValueText;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        Log.d(LOG_TAG, "----------");
        Log.d(LOG_TAG, "onCreate");

        setContentView(R.layout.activity_main);

        messageEditText = findViewById(R.id.editText_main);
        replyHeaderTxt = findViewById(R.id.reply_header_txt);
        replyValueText = findViewById(R.id.reply_txt);

        if(savedInstanceState != null) {
            boolean isVisible = savedInstanceState.getBoolean("reply_visible");

            if(isVisible) {
                replyHeaderTxt.setVisibility(View.VISIBLE);
                replyValueText.setText(savedInstanceState.getString("reply_text"));
                replyValueText.setVisibility(View.VISIBLE);
            }
        }

        Log.i(MainActivity.class.getName(), "started...");
    }

    @Override
    protected void onSaveInstanceState(Bundle outState) {
        super.onSaveInstanceState(outState);

        if(replyHeaderTxt.getVisibility() == View.VISIBLE) {
            outState.putBoolean("reply_visible", true);
            outState.putString("reply_text", replyValueText.getText().toString());
        }
    }

    @Override
    protected void onStart() {
        super.onStart();

        Log.d(LOG_TAG, "onStart");
    }

    @Override
    protected void onStop() {
        super.onStop();
        Log.d(LOG_TAG, "onStop");
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        Log.d(LOG_TAG, "onDestroy");
    }

    @Override
    protected void onPause() {
        super.onPause();
        Log.d(LOG_TAG, "onPause");
    }

    @Override
    protected void onResume() {
        super.onResume();
        Log.d(LOG_TAG, "onResume");
    }

    public void launchSecondActivity(View view) {

        Log.d(LOG_TAG, "Button clicked");

        String message = messageEditText.getText().toString();
        Intent intent = new Intent(this, ReplyActivity.class);
        intent.putExtra(EXTRA_MESSAGE, message);
        startActivityForResult(intent, TEXT_REQUEST);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if(requestCode == TEXT_REQUEST) {
            if(resultCode == RESULT_OK) {
                String reply = data.getStringExtra(ReplyActivity.EXTRA_REPLY);
                replyHeaderTxt.setVisibility(View.VISIBLE);
                replyValueText.setText(reply);
                replyValueText.setVisibility(View.VISIBLE);
            }
        }
    }
}
